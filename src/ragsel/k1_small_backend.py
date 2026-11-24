"""Owned local llama.cpp backend for the preselected smaller Qwen generator."""
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import Request, urlopen

from .k1_inference import LocalInference, InferenceResult
from .k1_protocol import file_sha256
from .k1_study_execution import ContractError

SMALL_SHA = '626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d'
SERVER_SHA = 'f087b3ae1ca31f2712ebd176875256e6079d3acb46597f31c7f5ff4085c9387d'
SMALL_MODEL = dict(name='qwen2.5-3b-instruct-q4_k_m-gguf', digest=SMALL_SHA)


def http(url, payload=None, timeout=600):
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None
    with urlopen(Request(url, data=data, headers={'Content-Type':'application/json'}), timeout=timeout) as response:
        return json.load(response)


@contextmanager
def owned_small_server(server, model, out, port=18089):
    server, model, out = Path(server).resolve(), Path(model).resolve(), Path(out).resolve()
    if file_sha256(server) != SERVER_SHA or file_sha256(model) != SMALL_SHA:
        raise ContractError('small-model/runtime hash differs from preselected assets')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', port)) == 0:
            raise ContractError('small-model port occupied; do not reuse an unverified server')
    out.mkdir(parents=True, exist_ok=True)
    stdout = (out / 'server.stdout.log').open('a', encoding='utf-8')
    stderr = (out / 'server.stderr.log').open('a', encoding='utf-8')
    process = subprocess.Popen([str(server), '--model', str(model), '--host','127.0.0.1', '--port',str(port),
        '--ctx-size','8192', '--parallel','1', '--n-gpu-layers','99', '--alias',SMALL_MODEL['name'], '--cache-ram','0'],
        cwd=server.parent, stdout=stdout, stderr=stderr, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    start = time.perf_counter()
    url = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + 180
        while True:
            if process.poll() is not None:
                raise ContractError('small server exited; see stderr log')
            try:
                if http(url + '/health', timeout=3).get('status') == 'ok':
                    break
            except Exception:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('small model loading timeout')
            time.sleep(.5)
        client = SmallInference(url, model)
        (out / 'environment.json').write_text(json.dumps(dict(model=SMALL_MODEL, digest_kind='gguf_file_sha256',
            server_sha256=SERVER_SHA, startup_seconds=time.perf_counter()-start, backend='llama.cpp b8967',
            props=http(url+'/props')), ensure_ascii=False, indent=2), encoding='utf-8')
        yield client
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=15)
        stdout.close(); stderr.close()


class SmallInference(LocalInference):
    backend = 'llama.cpp'
    expected_prompt_tokens = None

    def __init__(self, url, model_file):
        super().__init__(url, 600)
        self.model_file = Path(model_file)
        self.file_stat = (self.model_file.stat().st_size, self.model_file.stat().st_mtime_ns)

    def verify_model(self, model):
        if asdict(model) != SMALL_MODEL or self.file_stat != (self.model_file.stat().st_size, self.model_file.stat().st_mtime_ns):
            raise ContractError('small model identity/file changed')
        names = {row['id'] for row in self._get('/v1/models')['data']}
        if model.name not in names:
            raise ContractError('small-server alias differs')

    def chat(self, model, messages, options, recorder, call_id, role, thinking=None, schema=None, **kwargs):
        self.verify_model(model)
        mapped = [asdict(m) for m in messages]
        rendered = self._post('/apply-template', dict(messages=mapped))['prompt']
        actual = len(self._post('/tokenize', dict(content=rendered, add_special=False))['tokens'])
        if actual + options.output_tokens > options.context_tokens:
            raise ContractError('small-model full prompt overflow')
        if self.expected_prompt_tokens and actual != self.expected_prompt_tokens[call_id]:
            raise ContractError('actual small tokenizer differs from prepared prompt; regenerate policy before any test replies')
        body = dict(model=model.name, messages=mapped, temperature=options.temperature, seed=options.seed,
                    max_tokens=options.output_tokens, stream=False, cache_prompt=False)
        if schema is not None:
            body['response_format'] = dict(type='json_schema', json_schema=dict(name='answer', schema=schema, strict=True))
        def validate(response):
            choices = response.get('choices', [])
            if len(choices) != 1 or not isinstance(choices[0].get('message',{}).get('content'), str):
                raise ContractError('invalid OpenAI chat envelope')
            if response.get('usage',{}).get('prompt_tokens') != actual:
                raise ContractError('small-server evaluated prompt differs from preflight')
            if choices[0]['finish_reason'] == 'length': return 'truncated'
            if choices[0]['finish_reason'] != 'stop': return 'incomplete'
            return 'complete' if choices[0]['message']['content'].strip() else 'empty'
        response, status, elapsed = self._invoke(model, '/v1/chat/completions', body, recorder, call_id, role, validate)
        return InferenceResult(status, response['choices'][0]['message']['content'], elapsed, 0,
            response['usage']['prompt_tokens'], response['usage']['completion_tokens'], 0, 0)
