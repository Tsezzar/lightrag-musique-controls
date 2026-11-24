"""Prepare and resume fresh shared-block indices and native retrieval pools.

No final answers or gold readers. New graph directories only; moved native
workspaces are rejected. Main generation still needs a confirmatory protocol.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragsel.k1_protocol import file_sha256, config_fingerprint
from ragsel.k1_artifacts import freeze_json as freeze_manifest
from ragsel.k1_inference import LocalInference as _LocalInference


class AttributedInference(_LocalInference):
    corpus_id=None
    phase=None
    def _invoke(self,model,endpoint,body,recorder,call_id,role,validate):
        try:return super()._invoke(model,endpoint,body,recorder,call_id,role,validate)
        finally:
            row=recorder.con.execute('SELECT attempt FROM k1_calls WHERE call_id=? ORDER BY attempt DESC LIMIT 1',(call_id,)).fetchone()
            if row and self.corpus_id:
                recorder.con.execute('CREATE TABLE IF NOT EXISTS call_attribution(call_id TEXT,attempt INTEGER,corpus_id TEXT,phase TEXT,PRIMARY KEY(call_id,attempt))')
                recorder.con.execute('INSERT OR IGNORE INTO call_attribution VALUES(?,?,?,?)',(call_id,row[0],self.corpus_id,self.phase))
                recorder.con.commit()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_input(relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("portable input path required")
    path = (ROOT / relative).resolve()
    if ".." in Path(relative).parts or not path.is_relative_to(ROOT):
        raise ValueError("input path escapes package")
    return path


def checked_json(path, expected):
    if file_sha256(path) != expected:
        raise ValueError("frozen input hash mismatch")
    return read(path)


def prepare(args):
    assembly_path = ROOT / "results/k1_inputs/assembly-v2/manifest.json"
    resource_path = ROOT / "docs/k1/CONTEXT_PLAN_RESOURCE_AUDIT.json"
    assembly, resource = read(assembly_path), read(resource_path)
    allowed = ({r["corpus_id"] for r in resource["eligible_unscored_test_corpora"]}
               if args.scope == "confirmatory" else
               {r["corpus_id"] for r in assembly["corpora"] if r["split"] in ("train", "calibration")})
    selected = allowed if args.corpora == ["all"] else set(args.corpora)
    if not selected or not selected <= allowed:
        raise ValueError("corpora outside declared scope")
    rows = []
    for row in sorted(assembly["corpora"], key=lambda r: r["corpus_id"]):
        if row["corpus_id"] in selected:
            rows.append({**{k: row[k] for k in ("corpus_id", "family", "split", "public_sha256", "questions", "blocks")},
                         "public_path": row["public_path"].replace("\\", "/")})
    frozen = read(ROOT / "results/k1_revision/main-v1/frozen.json")
    cfg = read(ROOT / "config/k1_revision.json")
    models = {role: frozen["models"][cfg["models"][role]] for role in ("generator", "extractor", "embedding")}
    paths = [assembly_path, resource_path, ROOT / "work/context_matrix/source_archive.json",
             ROOT / "config/k1_revision.json", ROOT / "config/k1_lightrag.json",
             ROOT / "config/k1_retrieval.json", ROOT / "data/k1_tokenizers/qwen_tokenizer.json",
             ROOT / "results/k1_revision/main-v1/frozen.json", Path(__file__).resolve()]
    paths += [ROOT / "src/ragsel" / f for f in ("k1_lightrag.py", "k1_baselines.py", "k1_blocks.py",
        "k1_public.py", "k1_inference.py", "k1_protocol.py", "k1_comparison.py", "k1_artifacts.py")]
    paths += sorted((ROOT / "vendor/lightrag-source/lightrag").rglob("*.py"))
    native = read(ROOT / "config/k1_lightrag.json")
    native.update(query_max_tokens=64000, query_entity_tokens=16000, query_relation_tokens=16000)
    plan = dict(format="k1-retrieval-plan-v1", status="frozen", scope=args.scope,
        purpose="indexing and retrieval only; never final-answer generation", corpora=rows,
        generator_answers_authorized=False, confirmatory_analysis_protocol="not_yet_frozen",
        assembly_sha256=file_sha256(assembly_path), resource_audit_sha256=file_sha256(resource_path),
        archive_sha256=read(ROOT / "work/context_matrix/source_archive.json")["archive_sha256"],
        models=models, ollama=frozen["ollama"], native=native,
        retrieval=read(ROOT / "config/k1_retrieval.json"), seed=cfg["seed"],
        block_tokens=cfg["block_tokens"],
        tokenizer_path="data/k1_tokenizers/qwen_tokenizer.json",
        inputs={p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(set(paths))},
        pool_scope="native limited pool after internal token pruning, before external 4000-token packing",
        pool_budget_rationale="development old native6000 pools had fewer than 3000 source tokens; expand retrieval only, keep final evidence4000",
        reuse_policy="fresh native indices; resume only with identical input/code/model/workspace hashes")
    freeze_manifest(args.out / "plan.json", plan)
    print(json.dumps(dict(plan=str(args.out / "plan.json"), corpora=len(rows),
                          questions=sum(r["questions"] for r in rows), final_answers=0)))


def bind_execution(directory, plan_sha256):
    # This is a runtime record, deliberately separate from the portable plan.
    payload = dict(plan_sha256=plan_sha256, native_root=str(directory.resolve()),
                   workspace_rule="k1-SHA256(canonical absolute directory)[:16]")
    freeze_manifest(directory / "execution_identity.json", payload)


def validate_receipt(receipt, block, public_sha, workspace):
    expected = dict(block_id=block.block_id, text_sha256=config_fingerprint(block.text),
                    public_sha256=public_sha, workspace=workspace)
    if any(receipt.get(key) != value for key, value in expected.items()) or len(receipt.get("chunks", [])) != 1:
        raise ValueError("index receipt does not match source/workspace")
    return receipt["chunks"][0]


def native_snapshot(directory):
    names = {"graph_chunk_entity_relation.graphml", "kv_store_doc_status.json", "kv_store_entity_chunks.json",
             "kv_store_full_docs.json", "kv_store_full_entities.json", "kv_store_full_relations.json",
             "kv_store_relation_chunks.json", "kv_store_text_chunks.json", "vdb_chunks.json",
             "vdb_entities.json", "vdb_relationships.json"}
    files = {p.relative_to(directory).as_posix(): file_sha256(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name in names}
    if not files:
        raise ValueError("native graph has no persistent index files")
    return files


def sealed(path):
    seal = path.with_suffix(".sha256.json")
    if not seal.exists() or read(seal) != dict(sha256=file_sha256(path)):
        raise ValueError("saved pool is unsealed or changed; inspect interruption")
    return read(path)


def save_sealed(path, payload):
    freeze_manifest(path, payload)
    freeze_manifest(path.with_suffix(".sha256.json"), dict(sha256=file_sha256(path)))


@contextmanager
def operation(directory, phase, corpus_id):
    identity = uuid.uuid4().hex
    path = directory / "operations.jsonl"
    def emit(**values):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(operation_id=identity, phase=phase, corpus_id=corpus_id,
                utc=datetime.now(timezone.utc).isoformat(), **values)) + "\n")
    emit(event="start", status="running")
    started = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        emit(event="finish", status="failed", wall_seconds=time.perf_counter() - started,
             error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        emit(event="finish", status="complete", wall_seconds=time.perf_counter() - started)


class Retrieval:
    def __init__(self, args):
        from tokenizers import Tokenizer
        from ragsel.k1_inference import CallRecorder, LocalInference, ModelRef
        self.args, self.out = args, args.plan.parent.resolve()
        self.plan = read(args.plan)
        if self.plan.get("status") != "frozen" or self.plan.get("format") != "k1-retrieval-plan-v1":
            raise ValueError("frozen retrieval plan required")
        for relative, expected in self.plan["inputs"].items():
            if file_sha256(resolve_input(relative)) != expected:
                raise ValueError(f"frozen code/input changed: {relative}")
        bind_execution(self.out, file_sha256(args.plan))
        self.api = AttributedInference(args.url, self.plan["native"]["timeout_seconds"])
        if self.api.version() != self.plan["ollama"]:
            raise ValueError("Ollama runtime version differs from frozen plan")
        self.refs = {role: ModelRef(**value) for role, value in self.plan["models"].items()}
        for model in self.refs.values():
            self.api.verify_model(model)
            description = self.api.describe(model)
            if description.get("remote_host") or description.get("remote_model"):
                raise ValueError("remote/cloud models are not permitted")
        description = self.api.describe(self.refs["embedding"])
        dimensions = [v for k, v in description["model_info"].items() if k.endswith(".embedding_length")]
        if len(dimensions) != 1:
            raise ValueError("embedding dimension is ambiguous")
        self.dimension = dimensions[0]
        import lightrag
        if not Path(lightrag.__file__).resolve().is_relative_to(ROOT / "vendor/lightrag-source"):
            raise ValueError("the frozen vendor LightRAG implementation must take precedence")
        freeze_manifest(self.out / "environment.json", dict(python=platform.python_version(),
            platform=platform.platform(), ollama=self.api.version(),
            lightrag_relative_path=Path(lightrag.__file__).resolve().relative_to(ROOT).as_posix(),
            packages={name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "scikit-learn", "rank-bm25", "tokenizers", "requests", "lightrag-hku")}))
        self.tokenizer = Tokenizer.from_file(str(resolve_input(self.plan["tokenizer_path"])))
        self.con = sqlite3.connect(self.out / "calls.sqlite")
        self.recorder = CallRecorder(self.con)

    def state(self, status, **data):
        path = self.out / "status.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(status=status, **data), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        print(json.dumps(dict(status=status, **data)), flush=True)

    def check_retry(self, call_id):
        prior = self.recorder.con.execute('SELECT status FROM k1_calls WHERE call_id=? ORDER BY attempt DESC LIMIT 1',(call_id,)).fetchone()
        if prior and prior[0] != "complete" and not self.args.retry_failed:
            raise RuntimeError("failed/interrupted call requires inspection and --retry-failed")

    def embedding(self, path, texts, call_id):
        if path.exists():
            payload = read(path)
            if payload["texts_sha256"] != config_fingerprint(texts) or payload["model"] != asdict(self.refs["embedding"]):
                raise ValueError("cached embedding provenance differs")
            if payload["vectors_sha256"] != config_fingerprint(payload["vectors"]):
                raise ValueError("cached embedding vector hash mismatch")
            return payload["vectors"]
        self.check_retry(call_id)
        result = self.api.embed(self.refs["embedding"], texts, self.dimension, self.recorder, call_id, "matrix/embedding")
        freeze_manifest(path, dict(texts_sha256=config_fingerprint(texts), model=asdict(self.refs["embedding"]),
            vectors=result.vectors, vectors_sha256=config_fingerprint(result.vectors), wall_seconds=result.wall_seconds))
        return result.vectors

    def baselines(self, corpus):
        from ragsel.k1_baselines import BM25Index, VectorIndex
        index_start=time.perf_counter()
        vectors = []
        old_acquisition=0.;new_acquisition=0.
        self.api.corpus_id=corpus.corpus_id;self.api.phase='vector_index'
        batch = self.plan["native"]["embedding_batch_size"]
        for offset in range(0, len(corpus.blocks), batch):
            texts = tuple(b.text for b in corpus.blocks[offset:offset + batch])
            embedding_path=self.out / "index" / corpus.corpus_id / f"embedding-{offset}.json"
            existed=embedding_path.exists()
            vectors.extend(self.embedding(embedding_path,
                texts, f"matrix/embed/index/{corpus.corpus_id}/{offset}"))
            acquisition=read(embedding_path)['wall_seconds'];old_acquisition+=acquisition
            if not existed:new_acquisition+=acquisition
        vector = VectorIndex(corpus.blocks, vectors)
        vector_index_seconds=time.perf_counter()-index_start+old_acquisition-new_acquisition
        start=time.perf_counter()
        lexical = BM25Index(corpus.blocks, **{k: self.plan["retrieval"]["bm25"][k] for k in ("k1", "b", "epsilon")})
        lexical_index_seconds=time.perf_counter()-start
        cost_path=self.out/'index'/corpus.corpus_id/'baseline_cost.json'
        if not cost_path.exists():freeze_manifest(cost_path,dict(vector_index_seconds=vector_index_seconds,bm25_index_seconds=lexical_index_seconds,
            note='first successful build; full failures/resumes are also in operations.jsonl and calls.sqlite'))
        depth = self.plan["retrieval"]["candidate_blocks"]
        self.api.phase='vector_query'
        results = {}
        for question in corpus.questions:
            qhash = config_fingerprint(question.qid)
            path = self.out / "baseline_pools" / corpus.corpus_id / (qhash + ".json")
            if path.exists():
                saved = sealed(path)
                if saved["question"] != asdict(question) or saved["corpus_id"] != corpus.corpus_id:
                    raise ValueError("saved baseline pool identity differs")
                results[question.qid] = path
                continue
            start = time.perf_counter()
            query = self.embedding(self.out / "index" / corpus.corpus_id / ("query-" + qhash + ".json"),
                (question.text,), f"matrix/embed/query/{corpus.corpus_id}/{question.qid}")
            vector_hits=vector.rank(query[0],depth);vector_query_seconds=time.perf_counter()-start
            start=time.perf_counter();lexical_hits=lexical.rank(question.text,depth);lexical_query_seconds=time.perf_counter()-start
            hits = dict(vector=vector_hits,bm25=lexical_hits)
            save_sealed(path, dict(corpus_id=corpus.corpus_id, question=asdict(question), hits=hits,
                retrieval_seconds=vector_query_seconds+lexical_query_seconds,
                per_method_seconds=dict(vector=vector_query_seconds,bm25=lexical_query_seconds),
                ranked_blocks={method: [asdict(next(b for b in corpus.blocks if b.block_id == hit["block_id"]))
                                       for hit in values] for method, values in hits.items()}))
            results[question.qid] = path
        self.api.unload(self.refs["embedding"])
        return results

    async def graph(self, corpus, record, baselines):
        from lightrag import QueryParam
        from lightrag.base import DocStatus
        from ragsel.k1_comparison import native_context
        from ragsel.k1_inference import GenerationOptions
        from ragsel.k1_lightrag import NativeBindings, make_native_rag
        directory = self.out / "graphs" / corpus.corpus_id
        self.api.corpus_id=corpus.corpus_id;self.api.phase='graph_index'
        index_start=time.perf_counter()
        new_receipt_seconds=0.
        binding = NativeBindings(self.api, self.recorder, self.refs["extractor"], self.refs["embedding"], self.dimension,
            GenerationOptions(self.plan["native"]["context_window"], self.plan["native"]["role_output_tokens"]["extract"], 0, self.plan["seed"]),
            self.plan["native"]["role_output_tokens"])
        rag = make_native_rag(directory, tuple(b.text for b in corpus.blocks), self.tokenizer, binding, self.plan["native"])
        workspace = "k1-" + config_fingerprint(str(directory.resolve()))[:16]
        completion = directory / "index_complete.json"
        if completion.exists() and native_snapshot(directory) != read(completion)["native_files"]:
            raise ValueError("persistent native index changed since completion")
        try:
            await rag.initialize_storages()
            chunks = {}
            titles = {document.source_id: document.title for document in corpus.documents}
            for index, block in enumerate(corpus.blocks):
                marker = directory / "receipts" / (block.block_id + ".json")
                if marker.exists():
                    chunk = validate_receipt(read(marker), block, record["public_sha256"], workspace)
                else:
                    statuses = await rag.doc_status.get_by_ids([block.block_id])
                    if statuses and statuses[0] and statuses[0].get("status") != DocStatus.PROCESSED and not self.args.retry_failed:
                        raise RuntimeError("incomplete graph block requires inspection and --retry-failed")
                    start = time.perf_counter()
                    # Recover processed-without-receipt after an interruption without reinserting.
                    if not statuses or not statuses[0] or statuses[0].get("status") != DocStatus.PROCESSED:
                        await rag.ainsert(block.text, ids=[block.block_id], file_paths=[titles[block.source_id] + "::" + block.block_id])
                    statuses = await rag.doc_status.get_by_ids([block.block_id])
                    if len(statuses) != 1 or not statuses[0] or statuses[0]["status"] != DocStatus.PROCESSED:
                        raise RuntimeError("native graph block did not reach PROCESSED")
                    receipt = dict(block_id=block.block_id, text_sha256=config_fingerprint(block.text),
                        public_sha256=record["public_sha256"], workspace=workspace,
                        chunks=statuses[0].get("chunks_list", []), wall_seconds=time.perf_counter() - start)
                    receipt['native_status_metadata']=statuses[0].get('metadata',{})
                    new_receipt_seconds+=receipt['wall_seconds']
                    chunk = validate_receipt(receipt, block, record["public_sha256"], workspace)
                    freeze_manifest(marker, receipt)
                saved = await rag.text_chunks.get_by_ids([chunk])
                if len(saved) != 1 or not saved[0] or saved[0]["content"] != block.text:
                    raise ValueError("native stored chunk differs from exact public source")
                chunks[chunk] = block
                self.state("indexing", corpus_id=corpus.corpus_id, blocks=index + 1, total_blocks=len(corpus.blocks))
            freeze_manifest(completion, dict(public_sha256=record["public_sha256"], workspace=workspace,
                blocks=len(corpus.blocks), native_files=native_snapshot(directory),
                receipts={p.name: file_sha256(p) for p in sorted((directory / "receipts").glob("*.json"))}))
            if not (directory/'index_cost.json').exists():freeze_manifest(directory/'index_cost.json',dict(
                successful_stage_seconds=time.perf_counter()-index_start,
                receipt_seconds=sum(read(p)['wall_seconds'] for p in (directory/'receipts').glob('*.json')),
                canonical_index_seconds=sum(read(p)['wall_seconds'] for p in (directory/'receipts').glob('*.json'))+max(0,time.perf_counter()-index_start-new_receipt_seconds),
                native_truncation_documents=sum(bool(read(p).get('native_status_metadata',{}).get('llm_truncation')) for p in (directory/'receipts').glob('*.json')),
                note='technical index completion, not a claim of exhaustive semantic extraction; all attempts in raw logs'))
            cfg = self.plan["native"]
            self.api.phase='graph_retrieval'
            for question in corpus.questions:
                path = self.out / "pools" / corpus.corpus_id / (config_fingerprint(question.qid) + ".json")
                if path.exists():
                    saved = sealed(path)
                    if saved["question"] != asdict(question) or saved["public_sha256"] != record["public_sha256"]:
                        raise ValueError("saved native pool identity differs")
                    continue
                start = time.perf_counter()
                result = await rag.aquery_llm(question.text, QueryParam(mode=cfg["query_mode"], only_need_context=True,
                    enable_rerank=False, top_k=cfg["query_top_k"], chunk_top_k=cfg["query_chunk_top_k"],
                    max_total_tokens=cfg["query_max_tokens"], max_entity_tokens=cfg["query_entity_tokens"],
                    max_relation_tokens=cfg["query_relation_tokens"], include_references=True))
                raw = native_context(result)
                ranked = []
                for chunk in result.get("data", {}).get("chunks", []):
                    block = chunks[chunk["chunk_id"]]
                    if chunk["content"] != block.text:
                        raise ValueError("native returned altered source")
                    ranked.append(asdict(block))
                if len({b["block_id"] for b in ranked}) != len(ranked):
                    raise ValueError("native pool contains duplicate source blocks")
                baseline = read(baselines[question.qid])
                save_sealed(path, dict(corpus_id=corpus.corpus_id, family=corpus.family, split=corpus.split,
                    public_sha256=record["public_sha256"], question=asdict(question), native_result=result,
                    native_context=raw, ranked_blocks={**baseline["ranked_blocks"], "lightrag": ranked},
                    baseline_pool_sha256=file_sha256(baselines[question.qid]), retrieval_seconds=time.perf_counter() - start,
                    per_method_seconds={**baseline.get('per_method_seconds',{}),'lightrag':time.perf_counter()-start},
                    native_processing_info=result.get("metadata", {}).get("processing_info"), pool_scope=self.plan["pool_scope"]))
                self.state("retrieval", corpus_id=corpus.corpus_id, qid=question.qid)
        finally:
            await rag.finalize_storages()
        self.api.unload(self.refs["extractor"])
        self.api.unload(self.refs["embedding"])

    async def run(self):
        from ragsel.k1_public import load_public_corpus
        try:
            for record in self.plan["corpora"]:
                corpus = load_public_corpus(resolve_input(record["public_path"]), record["public_sha256"], self.tokenizer, self.plan["block_tokens"])
                with operation(self.out, "baseline_pipeline", corpus.corpus_id):
                    baselines = self.baselines(corpus)
                with operation(self.out, "graph_index_and_retrieval_pipeline", corpus.corpus_id):
                    await self.graph(corpus, record, baselines)
            outputs = {p.relative_to(self.out).as_posix(): file_sha256(p) for p in sorted((self.out / "pools").rglob("*.json"))
                       if not p.name.endswith(".sha256.json")}
            if len(outputs) != sum(row["questions"] for row in self.plan["corpora"]):
                raise ValueError("retrieval pool count does not match plan")
            freeze_manifest(self.out / "pools_manifest.json", dict(plan_sha256=file_sha256(self.args.plan), pools=outputs,
                final_answers_generated=0, pool_scope=self.plan["pool_scope"]))
            self.state("complete", pools=len(outputs), final_answers=0)
        except BaseException as exc:
            self.state("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--scope", choices=("development", "confirmatory"), required=True)
    plan.add_argument("--corpora", nargs="+", default=["all"])
    plan.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("plan", type=Path)
    run.add_argument("--url", default="http://127.0.0.1:11434")
    run.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        prepare(args)
    else:
        sys.path.insert(0, str(ROOT / "vendor/lightrag-source"))
        lock = args.plan.parent / "RUNNING.lock"
        with lock.open("x", encoding="utf-8") as stream:
            stream.write("One process per plan. Inspect the old process before removing a stale lock.\n")
        try:
            asyncio.run(Retrieval(args).run())
        finally:
            lock.unlink()


if __name__ == "__main__":
    main()
