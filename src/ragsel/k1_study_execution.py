"""Scientific queue execution: retain all model replies, retry transport only.

An empty or token-budget-truncated model reply is an observed outcome, not an
excuse to sample a better answer. HTTP/incomplete/interrupted attempts are kept
and may be retried twice with identical requests. Missing answers remain null.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import time

from .k1_handoff import load_queue
from .k1_inference import CallRecorder, ModelRef, Message, GenerationOptions, LocalInference
from .k1_protocol import config_fingerprint, file_sha256, freeze_manifest

TERMINAL = {'complete', 'empty', 'truncated'}


class ContractError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def latest_rows(recorder):
    return {row['call_id']: row for row in recorder.rows()}


class StrictOllama(LocalInference):
    backend = 'ollama'
    expected_prompt_tokens = None

    def chat(self, model, messages, options, recorder, call_id, role, **kwargs):
        result = super().chat(model, messages, options, recorder, call_id, role, **kwargs)
        expected = self.expected_prompt_tokens.get(call_id) if self.expected_prompt_tokens else None
        if result.status in TERMINAL and expected is not None and result.prompt_tokens != expected:
            attempt, response = recorder.con.execute(
                'SELECT attempt,response_json FROM k1_calls WHERE call_id=? ORDER BY attempt DESC LIMIT 1', (call_id,)).fetchone()
            recorder.finish(call_id, attempt, 'invalid_prompt_count', result.wall_seconds,
                            json.loads(response), f'expected {expected}, observed {result.prompt_tokens}')
            raise ContractError(f'{call_id}: prompt token count differs ({result.prompt_tokens} vs {expected}); no tolerance fitted')
        return result


def run_scientific_queue(path, out, client, *, limit=None, maximum_transport_retries=2):
    path, out = Path(path), Path(out)
    queue = load_queue(path)
    protocol = json.loads((path.parent / 'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('backend', 'ollama') != client.backend:
        raise ContractError('queue/client backend mismatch')
    model = ModelRef(**queue['model'])
    client.verify_model(model)
    out.mkdir(parents=True, exist_ok=True)
    freeze_manifest(out / 'identity.json', dict(queue_sha256=file_sha256(path)))
    for source in (path, path.parent / 'protocol.json'):
        target = out / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ContractError('frozen queue/protocol changed on resume')
        if not target.exists():
            target.write_bytes(source.read_bytes())
    manifest = json.loads((path.parent / 'contexts_manifest.json').read_text(encoding='utf-8'))
    expected = {row['id']: row['prompt_tokens_offline'] for row in manifest['entries']}
    if set(expected) != {job['id'] for job in queue['jobs']}:
        raise ContractError('prompt ledger does not match queue population')
    client.expected_prompt_tokens = expected
    attempted = 0
    started = time.perf_counter()
    with sqlite3.connect(out / 'calls.sqlite') as connection:
        recorder = CallRecorder(connection)
        previous = latest_rows(recorder)
        foreign = set(previous) - {job['id'] for job in queue['jobs']}
        if foreign:
            raise ContractError('foreign condition in raw call database')
        try:
            for job in queue['jobs']:
                last = previous.get(job['id'])
                if last and last['status'] in TERMINAL:
                    continue
                if last and last['status'] == 'invalid_prompt_count':
                    raise ContractError('unresolved prompt-token contract failure')
                if limit is not None and attempted >= limit:
                    break
                while True:
                    count = connection.execute('SELECT count(*) FROM k1_calls WHERE call_id=?', (job['id'],)).fetchone()[0]
                    if count > maximum_transport_retries:
                        raise RuntimeError(f'{job["id"]}: transport retry ceiling reached; inspect all attempts')
                    try:
                        result = client.chat(model, tuple(Message(**m) for m in job['messages']),
                            GenerationOptions(**queue['options']), recorder, job['id'], 'generator',
                            thinking=job.get('thinking'), schema=job.get('schema'))
                    except ContractError:
                        raise
                    except (ValueError, TypeError):
                        raise
                    except Exception:
                        if count >= maximum_transport_retries:
                            raise
                        continue
                    if result.status in TERMINAL:
                        break
                    if result.status == 'invalid_prompt_count':
                        raise ContractError('invalid prompt count')
                    if count >= maximum_transport_retries:
                        raise RuntimeError('incomplete response after fixed retry ceiling')
                attempted += 1
                if attempted % 10 == 0:
                    print(json.dumps(dict(stage='generation', queue=path.parent.name,
                                          new_terminal_replies=attempted, total=len(queue['jobs']))), flush=True)
        finally:
            rows = recorder.rows()
            with (out / 'answers.jsonl.tmp').open('w', encoding='utf-8') as stream:
                metadata = {job['id']: job['metadata'] for job in queue['jobs']}
                for row in rows:
                    payload = {**row, 'metadata': metadata[row['call_id']],
                        'response': json.loads(row['response_json']) if row['response_json'] else None}
                    stream.write(json.dumps(payload, ensure_ascii=False) + '\n')
            (out / 'answers.jsonl.tmp').replace(out / 'answers.jsonl')
            latest = {row['call_id']: row for row in rows}
            counts = {status: sum(row['status'] == status for row in latest.values()) for status in sorted({r['status'] for r in rows})}
            atomic_json(out / 'status.json', dict(total=len(queue['jobs']), terminal_replies=sum(r['status'] in TERMINAL for r in latest.values()),
                raw_status_counts=counts, attempts=len(rows), elapsed_this_invocation_seconds=time.perf_counter()-started,
                quality_evaluated=False))
    return json.loads((out / 'status.json').read_text(encoding='utf-8'))
