"""Resumable public-input development comparison. Reference files are never read here."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .k1_assembly import StudyCorpus
from .k1_baselines import answer_messages, answer_schema
from .k1_blocks import token_count
from .k1_inference import CallRecorder, GenerationOptions, LocalInference, ModelRef
from .k1_protocol import config_fingerprint, file_sha256, freeze_manifest


def development_records(records: list[dict], sources) -> list[dict]:
    """Complete declared development population; no outcome-dependent selection."""
    selected = [record for record in records if record["split"] in ("train", "calibration")]
    if set(record["family"] for record in selected) != set(sources):
        raise ValueError("development source families differ from protocol")
    if len({record["corpus_id"] for record in selected}) != len(selected):
        raise ValueError("duplicate development corpus identity")
    grouped = {source: sorted([r for r in selected if r["family"] == source], key=lambda r: r["corpus_id"])
               for source in sources}
    # Interleave sources to avoid putting every cold-start/order effect in one family.
    result = []
    while any(grouped.values()):
        for source in sources:
            if grouped[source]:
                result.append(grouped[source].pop(0))
    return result


def native_context(result: dict) -> str:
    response = result["llm_response"]
    if response["llm_generated"] is not False or response["is_streaming"] is not False:
        raise RuntimeError("native context retrieval unexpectedly generated or streamed an answer")
    if result["status"] == "failure" and result["metadata"].get("failure_reason") == "no_results":
        return ""
    if result["status"] != "success":
        raise RuntimeError(f"native retrieval failed: {result.get('message')}")
    if not isinstance(response["content"], str):
        raise ValueError("native context is not text")
    return response["content"]


class ComparisonRunner:
    def __init__(self, api: LocalInference, recorder: CallRecorder, directory: Path, tokenizer,
                 context_tokens: int, generator: ModelRef, options: GenerationOptions, before_call,
                 thinking=None):
        self.api, self.recorder, self.directory, self.tokenizer = api, recorder, directory, tokenizer
        self.context_tokens, self.generator, self.options = context_tokens, generator, options
        self.before_call, self.thinking = before_call, thinking

    @staticmethod
    def validate_scope(corpus: StudyCorpus):
        if corpus.split not in ("train", "calibration"):
            raise ValueError("test gate required; development runner cannot access test corpora")

    def path(self, kind: str, corpus_id: str, method: str, qid: str) -> Path:
        if method not in ("bm25", "vector", "lightrag") or Path(corpus_id).name != corpus_id:
            raise ValueError("invalid public comparison identity")
        return self.directory / kind / corpus_id / method / (config_fingerprint(qid) + ".json")

    def save_context(self, corpus, method, qid, text, retrieval_seconds, pieces, detail):
        self.validate_scope(corpus)
        count = token_count(self.tokenizer, text)
        if count > self.context_tokens:
            raise ValueError("delivered context exceeds the frozen generator token budget")
        payload = dict(corpus_id=corpus.corpus_id, family=corpus.family, split=corpus.split,
                       method=method, qid=qid, text=text, context_tokens=count,
                       retrieval_seconds=retrieval_seconds, pieces=pieces, detail=detail)
        freeze_manifest(self.path("contexts", corpus.corpus_id, method, qid), payload)
        return payload

    def generate(self, corpus, method, question):
        self.validate_scope(corpus)
        context_path = self.path("contexts", corpus.corpus_id, method, question.qid)
        context = json.loads(context_path.read_text(encoding="utf-8"))
        if (context["qid"] != question.qid or context["corpus_id"] != corpus.corpus_id
                or context["method"] != method or context["split"] != corpus.split):
            raise ValueError("context identity differs from the requested trial")
        if token_count(self.tokenizer, context["text"]) > self.context_tokens:
            raise ValueError("stored context exceeds token budget")
        messages = answer_messages(corpus.family, question.text, context["text"])
        identity = config_fingerprint(dict(model=asdict(self.generator), options=asdict(self.options),
                    messages=[asdict(m) for m in messages], schema=answer_schema(corpus.family), thinking=self.thinking,
                    context_sha256=file_sha256(context_path)))
        output_path = self.path("answers", corpus.corpus_id, method, question.qid)
        if output_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            if result["request_identity"] != identity:
                raise ValueError("frozen answer request changed on resume")
            return result
        self.before_call()
        out = self.api.chat(self.generator, messages, self.options, self.recorder,
                            f"answer/{corpus.corpus_id}/{method}/{question.qid}", "main/generator",
                            thinking=self.thinking, schema=answer_schema(corpus.family))
        result = dict(corpus_id=corpus.corpus_id, family=corpus.family, split=corpus.split,
                      method=method, qid=question.qid, generator=asdict(self.generator), request_identity=identity,
                      context_sha256=file_sha256(context_path), context_tokens=context["context_tokens"],
                      status=out.status, answer=out.content, generation_seconds=out.wall_seconds,
                      retrieval_seconds=context["retrieval_seconds"], prompt_tokens=out.prompt_tokens,
                      output_tokens=out.output_tokens)
        freeze_manifest(output_path, result)
        return result
