"""Local-only, uncached inference with a durable record of every attempt."""
from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, replace
from urllib.parse import urlsplit

import requests

from .k1_protocol import config_fingerprint


@dataclass(frozen=True)
class ModelRef:
    name: str
    digest: str


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class GenerationOptions:
    context_tokens: int
    output_tokens: int
    temperature: float
    seed: int

    def payload(self) -> dict:
        if self.context_tokens <= 0 or self.output_tokens <= 0 or self.temperature < 0:
            raise ValueError("invalid generation limits")
        return dict(num_ctx=self.context_tokens, num_predict=self.output_tokens,
                    temperature=self.temperature, seed=self.seed)


@dataclass(frozen=True)
class InferenceResult:
    status: str
    content: str
    wall_seconds: float
    load_seconds: float
    prompt_tokens: int
    output_tokens: int
    prompt_seconds: float
    output_seconds: float

    @property
    def tokens_per_second(self) -> float | None:
        return self.output_tokens / self.output_seconds if self.output_seconds > 0 else None


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    wall_seconds: float
    prompt_tokens: int


class CallRecorder:
    """Append attempts, including failed and interrupted calls; never act as a cache."""

    def __init__(self, connection: sqlite3.Connection):
        self.con = connection
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS k1_calls(
                call_id TEXT, attempt INTEGER, role TEXT, model TEXT, model_digest TEXT,
                request_sha256 TEXT, request_json TEXT, started_utc TEXT,
                status TEXT, wall_seconds REAL, response_json TEXT, error TEXT,
                PRIMARY KEY(call_id,attempt));
        """)

    def begin(self, call_id: str, role: str, model: ModelRef, body: dict) -> int:
        identity = config_fingerprint(dict(model_digest=model.digest, role=role, body=body))
        previous = self.con.execute("SELECT attempt,request_sha256 FROM k1_calls WHERE call_id=? ORDER BY attempt",
                                    (call_id,)).fetchall()
        if any(row[1] != identity for row in previous):
            raise ValueError(f"call identity changed for {call_id}")
        attempt = previous[-1][0] + 1 if previous else 0
        self.con.execute("INSERT INTO k1_calls VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                         (call_id, attempt, role, model.name, model.digest, identity,
                          json.dumps(body, ensure_ascii=False), dt.datetime.now(dt.timezone.utc).isoformat(),
                          "running", None, None, None))
        self.con.commit()
        return attempt

    def finish(self, call_id: str, attempt: int, status: str, seconds: float,
               response: dict | None = None, error: str | None = None) -> None:
        self.con.execute("UPDATE k1_calls SET status=?,wall_seconds=?,response_json=?,error=? WHERE call_id=? AND attempt=?",
                         (status, seconds, json.dumps(response, ensure_ascii=False) if response is not None else None,
                          error, call_id, attempt))
        self.con.commit()

    def rows(self) -> list[dict]:
        cursor = self.con.execute("SELECT * FROM k1_calls ORDER BY rowid")
        names = [col[0] for col in cursor.description]
        return [dict(zip(names, row)) for row in cursor]


class LocalInference:
    """Only loopback Ollama endpoints. No pull/install, inference cache, or silent retries."""

    def __init__(self, url: str, timeout: float, session=None):
        parts = urlsplit(url)
        if (parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1")
                or parts.username or parts.password or parts.path not in ("", "/")
                or parts.query or parts.fragment):
            raise ValueError("an unambiguous loopback HTTP URL is required")
        if timeout <= 0:
            raise ValueError("positive inference timeout required")
        self.url, self.timeout = url.rstrip("/"), timeout
        self.session = session if session is not None else requests.Session()
        self.session.trust_env = False  # local requests must not use an external proxy

    def _get(self, endpoint: str) -> dict:
        out = self.session.get(self.url + endpoint, timeout=(5, self.timeout))
        out.raise_for_status()
        return out.json()

    def _post(self, endpoint: str, payload: dict) -> dict:
        out = self.session.post(self.url + endpoint, json=payload, timeout=(5, self.timeout))
        out.raise_for_status()
        return out.json()

    def version(self) -> dict:
        return self._get("/api/version")

    def installed(self) -> dict[str, ModelRef]:
        return {m["name"]: ModelRef(m["name"], m["digest"]) for m in self._get("/api/tags")["models"]}

    def running(self) -> dict:
        return self._get("/api/ps")

    def describe(self, model: ModelRef) -> dict:
        return self._post("/api/show", dict(model=model.name))

    def verify_model(self, model: ModelRef) -> None:
        if self.installed().get(model.name) != model:
            raise ValueError(f"model identity changed or missing: {model.name}")

    def unload(self, model: ModelRef) -> dict:
        return self._post("/api/generate", dict(model=model.name, keep_alive=0, stream=False))

    def chat(self, model: ModelRef, messages: tuple[Message, ...], options: GenerationOptions,
             recorder: CallRecorder, call_id: str, role: str, thinking: bool | str | None = None,
             schema: dict | None = None, keep_alive: str = "5m") -> InferenceResult:
        body = dict(model=model.name, messages=[asdict(m) for m in messages], options=options.payload(),
                    stream=False, keep_alive=keep_alive)
        if thinking is not None:
            body["think"] = thinking
        if schema is not None:
            body["format"] = schema
        def validate(response):
            content = response.get("message", {}).get("content", "")
            if response.get("done") is not True:
                return "incomplete"
            if response.get("done_reason") == "length":
                return "truncated"
            if not isinstance(content, str) or not content.strip():
                return "empty"
            return "complete"
        response, status, elapsed = self._invoke(model, "/api/chat", body, recorder, call_id, role, validate)
        return InferenceResult(status, response.get("message", {}).get("content", ""), elapsed,
                               response.get("load_duration", 0) / 1e9, response.get("prompt_eval_count", 0),
                               response.get("eval_count", 0), response.get("prompt_eval_duration", 0) / 1e9,
                               response.get("eval_duration", 0) / 1e9)

    def embed(self, model: ModelRef, texts: tuple[str, ...], dimension: int, recorder: CallRecorder,
              call_id: str, role: str, keep_alive: str = "5m") -> EmbeddingResult:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts) or dimension <= 0:
            raise ValueError("nonempty embedding inputs and positive dimension required")
        body = dict(model=model.name, input=list(texts), truncate=False, keep_alive=keep_alive)
        def validate(response):
            vectors = response.get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                raise ValueError("embedding response count differs from input count")
            for vector in vectors:
                if (not isinstance(vector, list) or len(vector) != dimension
                        or any(type(value) not in (float, int) or not math.isfinite(value) for value in vector)
                        or not any(value != 0 for value in vector)):
                    raise ValueError("invalid embedding dimension, numeric value or zero vector")
            return "complete"
        response, _, elapsed = self._invoke(model, "/api/embed", body, recorder, call_id, role, validate)
        return EmbeddingResult(tuple(tuple(vector) for vector in response["embeddings"]), elapsed,
                               response.get("prompt_eval_count", 0))

    def probe_tokens(self, model: ModelRef, text: str, options: GenerationOptions,
                     recorder: CallRecorder, call_id: str) -> int:
        """One actual raw-prompt token-count probe; the intentionally capped output is not an answer."""
        if not isinstance(text, str) or not text:
            raise ValueError("nonempty raw token probe required")
        body = dict(model=model.name, prompt=text, raw=True, stream=False,
                    options=replace(options, output_tokens=1).payload())
        def validate(response):
            count = response.get("prompt_eval_count")
            if response.get("done") is not True or type(count) is not int or count < 0:
                raise ValueError("raw token probe did not return a completed input-token count")
            return "token_probe"
        response, _, _ = self._invoke(model, "/api/generate", body, recorder, call_id, "token_probe", validate)
        return response["prompt_eval_count"]

    def _invoke(self, model: ModelRef, endpoint: str, body: dict, recorder: CallRecorder,
                call_id: str, role: str, validate):
        """One identity check, attempt log and interruption policy for every real inference request."""
        attempt = recorder.begin(call_id, role, model, body)
        start = time.perf_counter()
        response = None
        try:
            self.verify_model(model)
            response = self._post(endpoint, body)
            self.verify_model(model)
            status = validate(response)
            elapsed = time.perf_counter() - start
            recorder.finish(call_id, attempt, status, elapsed, response)
            return response, status, elapsed
        except BaseException as error:
            # Log and propagate. No substitute answer, score, or retry is manufactured.
            status = "error" if isinstance(error, Exception) else "interrupted"
            recorder.finish(call_id, attempt, status, time.perf_counter() - start,
                            response=response, error=f"{type(error).__name__}: {error}")
            raise
