"""Thin bindings to the installed LightRAG; shared source blocks and recorded local inference."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .k1_inference import CallRecorder, GenerationOptions, LocalInference, Message, ModelRef
from .k1_protocol import config_fingerprint


class TokenizerAdapter:
    """Read-only HF tokenizer; no mutation during concurrent token counting."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __deepcopy__(self, memo):
        return self

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)


class ExactChunker:
    """Restore the registered original block after the vendor sanitizes its document input."""
    def __init__(self, texts: tuple[str, ...], sanitize, maximum_tokens: int):
        self.texts, self.maximum_tokens = {}, maximum_tokens
        for text in texts:
            key = sanitize(text)
            if key in self.texts and self.texts[key] != text:
                raise ValueError("ambiguous original blocks after vendor input sanitization")
            self.texts[key] = text

    def __deepcopy__(self, memo):
        return self

    def __call__(self, tokenizer, content, split_by_character, split_by_character_only, overlap, chunk_size):
        if content not in self.texts:
            raise ValueError("unregistered text reached the exact block chunker")
        original = self.texts[content]
        tokens = len(tokenizer.encode(original))
        if tokens > self.maximum_tokens:
            raise ValueError("registered block exceeds the shared token budget")
        return [dict(tokens=tokens, content=original, chunk_order_index=0)]


class NativeBindings:
    def __init__(self, api: LocalInference, recorder: CallRecorder, model: ModelRef, embedding: ModelRef,
                 embedding_dimension: int, options: GenerationOptions, role_limits: dict[str, int],
                 before_call=None, on_call=None):
        self.api, self.recorder, self.model, self.embedding = api, recorder, model, embedding
        self.embedding_dimension, self.options, self.role_limits = embedding_dimension, options, role_limits
        self.before_call, self.on_call = before_call, on_call
        self.sequence = recorder.con.execute('SELECT count(*) FROM k1_calls').fetchone()[0]

    def next_id(self, role):
        identity = f"native/{role}/{self.sequence:08d}"
        self.sequence += 1
        if self.before_call is not None:
            self.before_call()
        return identity

    def notify(self, role, out):
        if self.on_call is not None:
            self.on_call(dict(role=role, status=getattr(out, "status", "complete"),
                              wall_seconds=out.wall_seconds, output_tokens=getattr(out, "output_tokens", 0)))

    def llm(self, role: str):
        async def invoke(prompt, system_prompt=None, history_messages=None, **kwargs):
            messages = []
            if system_prompt is not None:
                messages.append(Message("system", system_prompt))
            messages.extend(Message(m["role"], m["content"]) for m in (history_messages or []))
            messages.append(Message("user", prompt))
            if any(not isinstance(message.content, str) for message in messages):
                raise ValueError("native text-only binding received nontext content")
            fmt = kwargs.get("response_format")
            if fmt is not None and fmt != {"type": "json_object"}:
                raise ValueError("unsupported native structured output format")
            limit = min(self.role_limits[role], kwargs.get("max_tokens") or self.role_limits[role])
            out = self.api.chat(self.model, tuple(messages), replace(self.options, output_tokens=limit),
                self.recorder, self.next_id(role), f"lightrag/{role}", thinking=False,
                schema={"type": "object"} if fmt else None)
            self.notify(role, out)
            if out.status == "truncated":
                from lightrag.utils import TruncatedResponse
                return TruncatedResponse(out.content)
            if out.status != "complete":
                raise RuntimeError(f"native LightRAG LLM response is {out.status}")
            return out.content
        return invoke

    async def embed(self, texts):
        out = self.api.embed(self.embedding, tuple(texts), self.embedding_dimension, self.recorder,
                             self.next_id("embedding"), "lightrag/embedding")
        self.notify("embedding", out)
        matrix = np.asarray(out.vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("embedding normalization is not finite and nonzero")
        return matrix / norms


def make_native_rag(directory, texts: tuple[str, ...], tokenizer, binding: NativeBindings, cfg: dict):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc, Tokenizer, sanitize_text_for_encoding
    async def embed(texts):
        # Functions survive dataclasses.asdict/deepcopy without trying to copy
        # the bound method's live HTTP session and SQLite connection.
        return await binding.embed(texts)
    return LightRAG(working_dir=str(directory), workspace="k1-" + config_fingerprint(str(Path(directory).resolve()))[:16],
        tokenizer=Tokenizer("shared-generator-tokenizer", TokenizerAdapter(tokenizer)),
        chunking_func=ExactChunker(texts, sanitize_text_for_encoding, cfg["block_tokens"]),
        chunk_token_size=cfg["block_tokens"], chunk_overlap_token_size=0,
        llm_model_func=binding.llm("extract"), llm_model_name=binding.model.name,
        role_llm_configs={role: dict(func=binding.llm(role), max_async=1, timeout=cfg["timeout_seconds"])
                          for role in ("extract", "keyword", "query")},
        llm_model_max_async=1, max_parallel_insert=1, default_llm_timeout=cfg["timeout_seconds"],
        entity_extraction_use_json=cfg["json_extraction"],
        entity_extract_max_gleaning=cfg["gleaning"], entity_extract_max_entities=cfg["max_entities"],
        entity_extract_max_records=cfg["max_records"],
        summary_context_size=cfg["summary_context_tokens"], summary_max_tokens=cfg["summary_max_tokens"],
        summary_length_recommended=cfg["summary_recommended_tokens"],
        embedding_func=EmbeddingFunc(embedding_dim=binding.embedding_dimension, func=embed,
                                     max_token_size=cfg["embedding_token_limit"], model_name=binding.embedding.name),
        embedding_func_max_async=1,
        embedding_batch_num=cfg["embedding_batch_size"], default_embedding_timeout=cfg["timeout_seconds"],
        enable_llm_cache=False, enable_llm_cache_for_entity_extract=True,
        cosine_better_than_threshold=cfg["cosine_threshold"])
