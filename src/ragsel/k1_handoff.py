"""Portable generation-only bridge. Retrieval and the new experiment matrix are separate work."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import re
import sqlite3
import subprocess

from .k1_inference import CallRecorder, GenerationOptions, LocalInference, Message, ModelRef
from .k1_protocol import file_sha256


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_queue(path: Path) -> dict:
    path = Path(path)
    q = json.loads(path.read_text(encoding='utf-8'))
    if set(q) != {'format', 'protocol_sha256', 'model', 'options', 'jobs'} or q['format'] != 'k1-generation-queue-v1':
        raise ValueError('unsupported queue schema')
    protocol = path.parent / 'protocol.json'
    if file_sha256(protocol) != q['protocol_sha256']:
        raise ValueError('protocol identity changed')
    p = json.loads(protocol.read_text(encoding='utf-8'))
    if p.get('status') != 'frozen' or p.get('stage') not in ('development', 'confirmatory'):
        raise ValueError('protocol must be frozen and specify stage')
    model = q['model']
    if set(model) != {'name', 'digest'} or not isinstance(model['name'], str) or not model['name'].strip():
        raise ValueError('model identity required')
    if not isinstance(model['digest'], str) or not re.fullmatch(r'[0-9a-f]{64}', model['digest']):
        raise ValueError('full model digest required')
    options = q['options']
    if set(options) != {'context_tokens', 'output_tokens', 'temperature', 'seed'}:
        raise ValueError('generation options required')
    if any(type(options[k]) is not int for k in ('context_tokens', 'output_tokens', 'seed')):
        raise ValueError('integer generation limits and seed required')
    if type(options['temperature']) not in (int, float) or not math.isfinite(options['temperature']):
        raise ValueError('finite temperature required')
    GenerationOptions(**options).payload()
    if options['output_tokens'] >= options['context_tokens']:
        raise ValueError('output reserve must be smaller than context window')
    if not isinstance(q['jobs'], list) or not q['jobs']:
        raise ValueError('nonempty jobs required')
    ids = set()
    for job in q['jobs']:
        if not {'id', 'metadata', 'messages'} <= set(job) <= {'id', 'metadata', 'messages', 'schema', 'thinking'}:
            raise ValueError('unexpected job fields; labels must stay with evaluator')
        if not isinstance(job['id'], str) or not job['id'].strip() or job['id'] in ids:
            raise ValueError('unique nonempty job id required')
        ids.add(job['id'])
        if not isinstance(job['metadata'], dict) or not set(job['metadata']) <= {
                'corpus_id', 'question_id', 'arm', 'family', 'budget', 'graph_quota', 'context_sha256', 'seed'}:
            raise ValueError('unexpected metadata; labels must stay with evaluator')
        if not isinstance(job['messages'], list) or not job['messages']:
            raise ValueError('messages required')
        for message in job['messages']:
            if (set(message) != {'role', 'content'} or message['role'] not in ('system', 'user', 'assistant')
                    or not isinstance(message['content'], str) or not message['content'].strip()):
                raise ValueError('invalid message')
        if 'schema' in job and not isinstance(job['schema'], dict):
            raise ValueError('schema must be an object')
        if 'thinking' in job and type(job['thinking']) not in (bool, str):
            raise ValueError('invalid thinking option')
    return q


def export_results(out, queue, recorder):
    metadata = {job['id']: job['metadata'] for job in queue['jobs']}
    rows = recorder.rows()
    target = out / 'answers.jsonl.tmp'
    with target.open('w', encoding='utf-8') as stream:
        for row in rows:
            record = {k: v for k, v in row.items() if k not in ('request_json', 'response_json')}
            record.update(metadata=metadata[row['call_id']],
                          response=json.loads(row['response_json']) if row['response_json'] else None)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    target.replace(out / 'answers.jsonl')
    latest = {row['call_id']: row for row in rows}
    complete = sum(r['status'] == 'complete' for r in latest.values())
    write_json(out / 'status.json', dict(total=len(queue['jobs']), complete=complete,
                                        pending_or_failed=len(queue['jobs']) - complete,
                                        attempts=len(rows)))


def run_queue(path, out, client, *, retry_failed=False, limit=None):
    path, out = Path(path), Path(out)
    queue = load_queue(path)
    identity = file_sha256(path)
    if (out / 'identity.json').exists():
        if json.loads((out / 'identity.json').read_text()) != dict(queue_sha256=identity):
            raise ValueError('output identity differs from this queue')
    elif out.exists() and any(out.iterdir()):
        raise ValueError('output must be empty or have matching identity')
    model = ModelRef(**queue['model'])
    client.verify_model(model)
    description = client.describe(model)
    if description.get('remote_host') or description.get('remote_model'):
        raise ValueError('remote/cloud models are not permitted')
    out.mkdir(parents=True, exist_ok=True)
    lock = out / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write('One runner per output directory. Remove only after confirming it has stopped.\n')
    try:
        write_json(out / 'identity.json', dict(queue_sha256=identity))
        (out / 'queue.json').write_bytes(path.read_bytes())
        (out / 'protocol.json').write_bytes((path.parent / 'protocol.json').read_bytes())
        write_json(out / 'environment.json', dict(python=platform.python_version(), platform=platform.platform(),
                                                  ollama=client.version(), model=asdict(model), description=description))
        with sqlite3.connect(out / 'calls.sqlite') as connection:
            recorder = CallRecorder(connection)
            previous = {row['call_id']: row for row in recorder.rows()}
            attempted = 0
            try:
                for job in queue['jobs']:
                    last = previous.get(job['id'])
                    if last and last['status'] == 'complete':
                        continue
                    if last and not retry_failed:
                        raise RuntimeError(f"{job['id']}: previous {last['status']}; inspect, then use --retry-failed")
                    if limit is not None and attempted >= limit:
                        break
                    print(f"Running {job['id']}", flush=True)
                    result = client.chat(model, tuple(Message(**m) for m in job['messages']),
                                         GenerationOptions(**queue['options']), recorder, job['id'], 'generator',
                                         thinking=job.get('thinking'), schema=job.get('schema'))
                    attempted += 1
                    if result.status != 'complete':
                        raise RuntimeError(f"{job['id']}: {result.status}; inspect raw response before retry")
            finally:
                export_results(out, queue, recorder)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    doctor = sub.add_parser('doctor', help='Hardware and local model inventory; no inference')
    doctor.add_argument('--ollama', action='store_true')
    doctor.add_argument('--output', type=Path, required=True)
    validate = sub.add_parser('validate', help='Offline structure and protocol hash check')
    validate.add_argument('queue', type=Path)
    run = sub.add_parser('run', help='Run prepared prompts, resume completed answers')
    run.add_argument('queue', type=Path)
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--retry-failed', action='store_true')
    run.add_argument('--limit', type=int)
    run.add_argument('--timeout', type=float, default=600)
    run.add_argument('--url', default='http://127.0.0.1:11434')
    args = parser.parse_args()
    if args.command == 'doctor':
        info = dict(python=platform.python_version(), platform=platform.platform())
        try:
            probe = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'],
                                   capture_output=True, text=True, timeout=15)
            info['nvidia_smi'] = dict(returncode=probe.returncode, stdout=probe.stdout, stderr=probe.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            info['nvidia_smi'] = dict(error=str(exc))
        if args.ollama:
            client = LocalInference('http://127.0.0.1:11434', 15)
            info.update(ollama=client.version(), models=[asdict(m) for m in client.installed().values()])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, info)
    elif args.command == 'validate':
        queue = load_queue(args.queue)
        print(json.dumps(dict(valid=True, jobs=len(queue['jobs']), queue_sha256=file_sha256(args.queue))))
    else:
        if args.limit is not None and args.limit <= 0:
            parser.error('--limit must be positive')
        run_queue(args.queue, args.out, LocalInference(args.url, args.timeout),
                  retry_failed=args.retry_failed, limit=args.limit)


if __name__ == '__main__':
    main()
