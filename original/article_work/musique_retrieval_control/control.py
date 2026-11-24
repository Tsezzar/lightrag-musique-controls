"""MuSiQue ablation of link propagation over the same frozen retrieval records."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from scipy import sparse
from tokenizers import Tokenizer

from ragsel.k1_artifacts import freeze_json, workflow_lock
from ragsel.k1_blocks import DeliveredContext, DeliveredPiece, prefix_length, token_count
from ragsel.k1_context_matrix import render_qwen_chat, text_sha
from ragsel.k1_handoff import load_queue
from ragsel.k1_inference import Message
from ragsel.k1_protocol import config_fingerprint, file_sha256
from ragsel.k1_public import load_public_corpus
from ragsel.k1_statistics import interval_conclusion, stratified_comparison
from ragsel.k1_study_analysis import evaluate, read, read_conditions
from ragsel.k1_study_execution import atomic_json, run_scientific_queue
from ragsel.k1_study_policy import make_job

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
PREVIOUS = OUT.parent / "musique_format_control"
STRUCTURE = PROJECT / "experiment_v2/src/ragsel/structure.py"
PARAMETERS = PROJECT / "experiment_v2/config/structure_pilot.json"
QUEUE = OUT / "queue.json"
PROTOCOL = OUT / "protocol.json"
CONTEXTS = OUT / "contexts_manifest.json"
FLAT, GRAPH = "FLAT_RECORDS", "GRAPH_PPR"


def structure_module():
    spec = importlib.util.spec_from_file_location("structure_pilot", STRUCTURE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def strict_index(retrieval, corpus, record, engine):
    folder = retrieval / "graphs" / corpus.corpus_id
    receipt_path = folder / "index_complete.json"
    receipt = read(receipt_path)
    if receipt["public_sha256"] != record["public_sha256"] or receipt["blocks"] != len(corpus.blocks):
        raise ValueError("corpus and graph completion receipt differ")
    workspace = receipt["workspace"]
    names = ("vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json", "kv_store_text_chunks.json")
    stores = {}
    for name in names:
        path = folder / workspace / name
        if file_sha256(path) != receipt["native_files"][f"{workspace}/{name}"]:
            raise ValueError(f"frozen index changed: {path}")
        stores[name] = read(path)
    chunks = stores["vdb_chunks.json"]["data"]
    entities = stores["vdb_entities.json"]["data"]
    relations = stores["vdb_relationships.json"]["data"]
    dim = stores["vdb_chunks.json"]["embedding_dim"]
    if any(stores[name]["embedding_dim"] != dim for name in names[:3]):
        raise ValueError("stored embedding dimensions differ")
    text_chunks = stores["kv_store_text_chunks.json"]
    by_block = {b.block_id: b for b in corpus.blocks}
    blocks, chunk_index = [], {}
    for i, row in enumerate(chunks):
        uid = row["__id__"]
        block_id = row["full_doc_id"]
        block = by_block[block_id]
        marker = folder / "receipts" / f"{block_id}.json"
        if file_sha256(marker) != receipt["receipts"][marker.name]:
            raise ValueError("source mapping receipt changed")
        mapping = read(marker)
        if (mapping["block_id"] != block_id or mapping["chunks"] != [uid] or uid in chunk_index
                or row["content"] != block.text or text_chunks[uid]["content"] != block.text
                or mapping["text_sha256"] != config_fingerprint(block.text)):
            raise ValueError("native chunk is not the exact public text block")
        blocks.append(block)
        chunk_index[uid] = i
    if len(blocks) != len(corpus.blocks) or {b.block_id for b in blocks} != set(by_block):
        raise ValueError("native storage must cover all public blocks exactly once")
    names_to_index = {row["entity_name"]: i for i, row in enumerate(entities)}
    records = entities + relations
    if len(names_to_index) != len(entities) or len({r["__id__"] for r in records}) != len(records):
        raise ValueError("ambiguous entity or relationship identity")
    loc_r, loc_c, weights, edges = [], [], [], []
    missing = 0
    for i, row in enumerate(records):
        sources = sorted({chunk_index[key] for key in row["source_id"].split("<SEP>") if key})
        missing += not sources
        for j in sources:
            loc_r.append(i)
            loc_c.append(j)
            weights.append(1/len(sources))
    for i, row in enumerate(relations, start=len(entities)):
        for name in sorted({row["src_id"], row["tgt_id"]}):
            edges.append((names_to_index[name], i))
    if not edges:
        raise ValueError("no graph edges to test")
    provenance = sparse.csr_matrix((weights, (loc_r, loc_c)), shape=(len(records), len(chunks)))
    telemetry = dict(receipt_sha256=file_sha256(receipt_path), entities=len(entities), relations=len(relations),
                     edges=len(edges), blocks=len(blocks), missing_provenance_records=missing, dimension=dim)
    extraction = engine.FrozenExtraction(records, engine._vectors(records, dim), chunks, engine._vectors(chunks, dim),
                                         provenance, np.array(edges, dtype=np.int64), len(entities), {})
    return extraction, blocks, telemetry


def cached_queries(root, corpus, embedding):
    vectors, artifacts = [], {}
    for question in corpus.questions:
        path = root / "index" / corpus.corpus_id / f"query-{config_fingerprint(question.qid)}.json"
        receipt = read(path)
        if (receipt["model"] != embedding or receipt["texts_sha256"] != config_fingerprint((question.text,))
                or receipt["vectors_sha256"] != config_fingerprint(receipt["vectors"])
                or len(receipt["vectors"]) != 1):
            raise ValueError(f"question embedding identity differs: {question.qid}")
        vectors.append(receipt["vectors"][0])
        artifacts[question.qid] = file_sha256(path)
    return np.asarray(vectors, dtype=np.float32), artifacts


def ranked_prose(blocks, titles, tokenizer, budget):
    full, offsets, position = [], [], 0
    for block in blocks:
        heading = titles[block.source_id] + "\n"
        paragraph = heading + block.text + "\n\n"
        offsets.append((position + len(heading), block))
        full.append(paragraph)
        position += len(paragraph)
    prose = "".join(full)
    cut = prefix_length(tokenizer, prose, budget)
    text = prose[:cut]
    pieces = []
    for start, block in offsets:
        end = min(len(block.text), cut-start)
        if end > 0:
            pieces.append(DeliveredPiece(block.block_id, block.source_id, block.start, block.start+end, block.text[:end]))
    return DeliveredContext(text, tuple(pieces), token_count(tokenizer, text))


def pair_jobs(corpus, question, by_arm, titles, tokenizer, protocol, protocol_sha, retrieval_sha):
    cap = protocol["context_cap"]
    initial = {arm:ranked_prose(ranked, titles, tokenizer, cap) for arm, ranked in by_arm.items()}
    for limit in range(min(context.tokens for context in initial.values()), 0, -1):
        samples = []
        for arm, ranking in by_arm.items():
            ctx = initial[arm] if initial[arm].tokens == limit else ranked_prose(ranking, titles, tokenizer, limit)
            if not ctx.pieces:
                raise ValueError("usable evidence is missing")
            variant = dict(arm=arm, text=ctx.text, text_sha256=text_sha(ctx.text), context_tokens=ctx.tokens,
                           budget=cap, underfill_tokens=cap-ctx.tokens, graph_quota=0., source_limit=limit,
                           pieces=[asdict(p) for p in ctx.pieces], graph_rows=[],
                           top_sources=[b.source_id for b in ranking[:protocol["retrieval_metrics"]["top_k"]]],
                           ranking=[b.block_id for b in ranking])
            job, prompt = make_job(corpus, question, variant, protocol["model"], protocol["options"],
                                   protocol_sha, retrieval_sha)
            samples.append((job, variant, token_count(tokenizer, prompt)))
        if (all(variant["context_tokens"] == limit for _, variant, _ in samples)
                and samples[0][2] == samples[1][2]
                and samples[0][2] + protocol["options"]["output_tokens"] <= protocol["options"]["context_tokens"]):
            return samples
    raise ValueError(f"no exact plain-text pair for {corpus.corpus_id}/{question.qid}")


def prepare(root):
    previous = read(PREVIOUS / "protocol.json")
    queue = load_queue(PREVIOUS / "queue.json")
    study_path = root / "work/run5090/STUDY_FROZEN.json"
    study = read(study_path)
    retrieval = root / "work/run5090/main_retrieval"
    plan_path = retrieval / "plan.json"
    plan = read(plan_path)
    if file_sha256(plan_path) != read(retrieval / "pools_manifest.json")["plan_sha256"] or study["models"] != plan["models"]:
        raise ValueError("frozen retrieval manifest differs")
    stats = study["statistics"]
    cohort = [tuple(key) for key in previous["cohort"]]
    family = stats["primary_family"]
    corpora = {rec["corpus_id"]:rec for rec in plan["corpora"] if rec["family"] == family}
    if len(cohort) != stats["primary_questions"] or set(corpora) != {cid for cid, _ in cohort}:
        raise ValueError("corpus population differs from the previous trials")
    if queue["model"] != study["models"]["generator"]:
        raise ValueError("generation model differs from original")
    config = read(PARAMETERS)
    scoring = {key:config[key] for key in ("temperature", "seed_records", "damping", "max_iter", "tolerance",
                                           "dense_weight", "dense_depth", "swaps_per_edge")}
    # The stochastic walk contracts in L1 by damping. Bound the residual
    # from any two probability vectors, including a fully bipartite graph.
    if 0 < scoring["damping"] < 1:
        required = math.ceil(math.log(scoring["tolerance"]/2)/math.log(scoring["damping"])) + 1
        scoring["max_iter"] = max(scoring["max_iter"], required)
    scoring["rewire_seeds"] = []
    tok_path = root / "data/k1_tokenizers/qwen_tokenizer.json"
    if file_sha256(tok_path) != previous["input_sha256"]["tokenizer"]:
        raise ValueError("Qwen tokenizer identity differs")
    protocol = dict(
        status="frozen", stage="development", backend="ollama", family=family, cohort=cohort,
        scope="post-main within-corpus test of personalized PageRank using the frozen LightRAG extraction",
        statistics=stats, primary_metric=study["primary_metric"], options=queue["options"], model=queue["model"],
        embedding=study["models"]["embedding"], seed=study["seed"], error_policy=study["error_policy"],
        contrast=f"{GRAPH} minus {FLAT}", context_cap=study["evidence_budget"], scoring=scoring,
        retrieval_metrics=dict(top_k=6, secondary=["support recall in the first six blocks",
                                                  "support recall and all supports visible in delivered prose"]),
        design="Same frozen records, BGE-M3 vectors, question vector, provenance and dense score. "
               "Flat retains the cosine-based seed distribution. PPR propagates it over the natural bipartite "
               "entity/relationship-record incidence graph. Both project record mass onto source blocks and mix "
               "with the same dense block score. Stable ties follow stored block order. Only original titled prose "
               "is passed. Take the largest shared context within the frozen cap preserving both context and full "
               "prompt parity. No padding, selective exclusions, or access to references while preparing.",
        input_sha256=dict(study=file_sha256(study_path), plan=file_sha256(plan_path),
                          previous_protocol=file_sha256(PREVIOUS / "protocol.json"),
                          previous_queue=file_sha256(PREVIOUS / "queue.json"), tokenizer=file_sha256(tok_path),
                          structure=file_sha256(STRUCTURE), parameters=file_sha256(PARAMETERS),
                          assembly=study["inputs"]["assembly_sha256"], code=file_sha256(__file__)),
    )
    protocol_sha = config_fingerprint(protocol)
    engine, tokenizer = structure_module(), Tokenizer.from_file(str(tok_path))
    pairs, inputs = {}, {}
    for cid, record in sorted(corpora.items()):
        corpus = load_public_corpus(root / record["public_path"], record["public_sha256"], tokenizer,
                                    plan["block_tokens"])
        extraction, blocks, inventory = strict_index(retrieval, corpus, record, engine)
        query_vectors, queries = cached_queries(retrieval, corpus, protocol["embedding"])
        if query_vectors.shape != (len(corpus.questions), inventory["dimension"]) or not np.isfinite(query_vectors).all():
            raise ValueError("invalid frozen query embeddings")
        scores, diagnostics = engine.chunk_scores(extraction, query_vectors, scoring)
        if not diagnostics["graph_real"]["converged"] or not diagnostics["frozen"]["zero_edge_equals_flat"]:
            raise ValueError(f"PPR convergence or no-links identity failed: {cid}, {diagnostics}")
        inputs[cid] = dict(index=inventory, queries=queries, ppr=diagnostics["graph_real"])
        retrieval_sha = config_fingerprint(inputs[cid])
        titles = {d.source_id:d.title for d in corpus.documents}
        for column, question in enumerate(corpus.questions):
            by_arm = {
                arm: [blocks[i] for i in engine.top_indices(scores[name][:,column], len(blocks))]
                for arm, name in ((FLAT, "flat_records"), (GRAPH, "graph_real"))
            }
            pairs[cid, question.qid] = pair_jobs(corpus, question, by_arm, titles, tokenizer, protocol,
                                                protocol_sha, retrieval_sha)
        print(f"Prepared {len(pairs)} of {len(cohort)} matched pairs", flush=True)
    if sorted(pairs) != sorted(cohort):
        raise ValueError("prepared pair list differs from previous controls")
    jobs, entries = [], []
    for index, key in enumerate(sorted(pairs, key=config_fingerprint)):
        pair = pairs[key]
        if index % 2:
            pair.reverse()
        for job, variant, count in pair:
            path = OUT / "contexts" / (job["id"] + ".json")
            freeze_json(path, variant)
            jobs.append(job)
            entries.append(dict(id=job["id"], context_path=path.relative_to(OUT).as_posix(),
                                context_file_sha256=file_sha256(path), prompt_tokens_offline=count))
    freeze_json(PROTOCOL, protocol)
    if protocol_sha != file_sha256(PROTOCOL):
        raise ValueError("protocol serialization differs")
    freeze_json(CONTEXTS, dict(entries=entries))
    freeze_json(OUT / "inputs.json", inputs)
    freeze_json(QUEUE, dict(format=queue["format"], protocol_sha256=protocol_sha,
                            options=queue["options"], model=queue["model"], jobs=jobs))
    freeze_json(OUT / "freeze_receipt.json", frozen_hashes())
    _, pairs, _ = verify(root)
    print(f"Verified all {len(pairs)} context and prompt matches.", flush=True)


def frozen_hashes():
    return {path.stem:file_sha256(path) for path in (QUEUE, CONTEXTS, OUT / "inputs.json")}


def verify(root):
    if read(OUT / "freeze_receipt.json") != frozen_hashes():
        raise ValueError("frozen queue, contexts or inputs ledger changed")
    protocol, queue = read(PROTOCOL), load_queue(QUEUE)
    tok = root / "data/k1_tokenizers/qwen_tokenizer.json"
    for key, path in (("code", Path(__file__)), ("structure", STRUCTURE), ("parameters", PARAMETERS), ("tokenizer", tok)):
        if file_sha256(path) != protocol["input_sha256"][key]:
            raise ValueError(f"frozen code or tokenizer changed: {key}")
    tokenizer = Tokenizer.from_file(str(tok))
    entries = {row["id"]:row for row in read(CONTEXTS)["entries"]}
    pairs, paths = defaultdict(dict), {}
    if set(entries) != {job["id"] for job in queue["jobs"]}:
        raise ValueError("prompt ledger and generation queue differ")
    for job in queue["jobs"]:
        entry = entries[job["id"]]
        path = OUT / entry["context_path"]
        if file_sha256(path) != entry["context_file_sha256"]:
            raise ValueError("frozen context file changed")
        variant = read(path)
        meta = job["metadata"]
        key, arm = (meta["corpus_id"], meta["question_id"]), meta["arm"]
        prompt = render_qwen_chat(tuple(Message(**m) for m in job["messages"]))
        if (arm in pairs[key] or arm != variant["arm"] or text_sha(variant["text"]) != meta["context_sha256"]
                or variant["graph_rows"] or token_count(tokenizer, variant["text"]) != variant["context_tokens"]
                or token_count(tokenizer, prompt) != entry["prompt_tokens_offline"]
                or not job["messages"][1]["content"].startswith("Evidence:\n"+variant["text"]+"\n\nQuestion: ")):
            raise ValueError(f"context integrity changed: {key}")
        paths[job["id"]] = path
        pairs[key][arm] = job, variant, entry
    if sorted(pairs) != sorted(tuple(key) for key in protocol["cohort"]):
        raise ValueError("same complete cohort required")
    for key, arms in pairs.items():
        if set(arms) != {FLAT, GRAPH}:
            raise ValueError(f"missing arm: {key}")
        (_, flat, fe), (_, graph, ge) = arms[FLAT], arms[GRAPH]
        if (flat["context_tokens"] != graph["context_tokens"]
                or fe["prompt_tokens_offline"] != ge["prompt_tokens_offline"]
                or set(flat["ranking"]) != set(graph["ranking"])):
            raise ValueError(f"ranking candidates or lengths differ: {key}")
    return protocol, pairs, paths


def analyze(root):
    protocol, pairs, paths = verify(root)
    manifest_path = root / "results/k1_inputs/assembly-v2/manifest.json"
    if file_sha256(manifest_path) != protocol["input_sha256"]["assembly"]:
        raise ValueError("gold reference manifest changed")
    raw = read_conditions(QUEUE, OUT / "generation")
    expected = {j["id"]:e["prompt_tokens_offline"] for pair in pairs.values() for j, _, e in pair.values()}
    if set(expected) != {row["id"] for row in raw} or any(expected[r["id"]] != r["prompt_tokens"] for r in raw):
        raise ValueError("complete actual prompt parity required before reading references")
    report, scored, _ = evaluate(root, [(QUEUE, OUT / "generation")], OUT / "analysis",
                                 scope=protocol["scope"], contexts=paths)
    corpus_ids = sorted({cid for cid, _ in pairs})
    assembly = {r["corpus_id"]:r for r in read(manifest_path)["corpora"]}
    refs = {cid:read(root / assembly[cid]["gold_path"]) for cid in corpus_ids}
    metrics = defaultdict(dict)
    for row in scored:
        cid, qid, arm = row["corpus_id"], row["question_id"], row["arm"]
        support = set(refs[cid][qid]["support_ids"])
        context = pairs[cid, qid][arm][1]
        seen = {p["source_id"] for p in context["pieces"]}
        metrics[cid,qid][arm] = dict(
            f1=row["quality"], recall_at_6=len(support & set(context["top_sources"]))/len(support),
            delivered_recall=len(support & seen)/len(support), all_supports=float(support <= seen),
        )
    statistics = protocol["statistics"]
    grouped = defaultdict(list)
    for (cid, _), arms in metrics.items():
        for metric in arms[FLAT]:
            for arm in (FLAT, GRAPH):
                grouped[cid,arm,metric].append(arms[arm][metric])
    comparisons = {}
    for metric in next(iter(metrics.values()))[FLAT]:
        scores = {arm:{cid:mean(grouped[cid,arm,metric]) for cid in corpus_ids} for arm in (FLAT, GRAPH)}
        rows = [(cid, protocol["family"], scores[GRAPH][cid]-scores[FLAT][cid]) for cid in corpus_ids]
        result = stratified_comparison(rows, {protocol["family"]:statistics["primary_corpora"]},
                                       statistics["bootstrap_resamples"], statistics["alpha"], protocol["seed"])
        result.update(means={arm:mean(values.values()) for arm,values in scores.items()},
                      positive_corpora=sum(v>0 for _,_,v in rows), tied_corpora=sum(v==0 for _,_,v in rows),
                      corpus_deltas={cid:value for cid,_,value in rows})
        comparisons[metric] = result
    margin = statistics["practical_margin"]
    f1 = comparisons["f1"]
    f1.update(practical_margin=margin, conclusion=interval_conclusion(f1["interval"], margin))
    lengths = [arms[FLAT][1]["context_tokens"] for arms in pairs.values()]
    result = dict(
        scope=protocol["scope"], questions=len(pairs), corpora=len(corpus_ids), responses=len(raw), model=protocol["model"],
        equal_context_pairs=len(pairs), equal_full_prompt_pairs=len(pairs),
        differing_contexts=sum(arms[FLAT][1]["text"] != arms[GRAPH][1]["text"] for arms in pairs.values()),
        differing_top6=sum(arms[FLAT][1]["ranking"][:6] != arms[GRAPH][1]["ranking"][:6] for arms in pairs.values()),
        context_tokens=dict(mean=mean(lengths), minimum=min(lengths), maximum=max(lengths)),
        terminal_statuses=dict(Counter(row["raw_status"] for row in raw)),
        comparisons=comparisons, summary=report["summary"], sha256={
            **frozen_hashes(), "protocol":file_sha256(PROTOCOL), "calls":file_sha256(OUT / "generation/calls.sqlite")},
    )
    atomic_json(OUT / "result.json", result)
    print({k:v for k,v in result.items() if k != "comparisons"})
    for metric, value in comparisons.items():
        print(metric, {k:value[k] for k in ("means", "estimate", "interval", "positive_corpora", "tied_corpora")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "verify", "run", "analyze"])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        _, pairs, _ = verify(args.root)
        print(f"Verified {len(pairs)} pairs.")
        return
    with workflow_lock(OUT):
        if args.mode == "prepare":
            prepare(args.root)
        elif args.mode == "run":
            sys.path.insert(0, str(args.root / "work/run5090/shared_launcher"))
            from bootstrap_shared_gpu import server5090
            verify(args.root)
            with server5090(OUT / "backend") as (client, _):
                print(run_scientific_queue(QUEUE, OUT / "generation", client), flush=True)
        else:
            analyze(args.root)


if __name__ == "__main__":
    main()
