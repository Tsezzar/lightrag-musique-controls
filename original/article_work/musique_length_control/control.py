"""Length-matched MuSiQue follow-up using frozen retrieval and the study runner."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
import sys

from tokenizers import Tokenizer

from ragsel.k1_artifacts import freeze_json, workflow_lock
from ragsel.k1_blocks import render_context, token_count
from ragsel.k1_context_matrix import graph_rows, pack_graph, render_graph, text_sha, validate_ranked
from ragsel.k1_handoff import load_queue
from ragsel.k1_protocol import config_fingerprint, file_sha256
from ragsel.k1_public import load_public_corpus
from ragsel.k1_statistics import interval_conclusion, stratified_comparison
from ragsel.k1_study_analysis import evaluate, read, read_conditions
from ragsel.k1_study_execution import atomic_json, run_scientific_queue
from ragsel.k1_study_policy import make_job, rankings
from ragsel.k1_study_queues import MemoTokenizer

OUT = Path(__file__).resolve().parent
QUEUE = OUT / "queue.json"
PROTOCOL = OUT / "protocol.json"
CONTEXT_MANIFEST = OUT / "contexts_manifest.json"
TEXT, GRAPH = "TEXT_MATCHED", "GRAPH_MATCHED"


def evidence(arm, source, graph, rows, ceiling, text_limit, graph_limit, tokenizer):
    text = source.text + graph
    count = token_count(tokenizer, text)
    return dict(
        arm=arm, text=text, text_sha256=text_sha(text), source_text=source.text,
        graph_text=graph, context_tokens=count, source_tokens=source.tokens,
        graph_tokens=token_count(tokenizer, graph), graph_rows=rows,
        budget=ceiling, underfill_tokens=ceiling-count, source_limit=text_limit,
        graph_limit=graph_limit, graph_quota=.25 if rows else 0.,
        pieces=[asdict(piece) for piece in source.pieces],
    )


def matched_pair(corpus, question, pool, tokenizer, study, protocol, pool_sha):
    blocks = rankings(pool)["lightrag"]
    validate_ranked(corpus, blocks)
    titles = {doc.source_id: doc.title for doc in corpus.documents}
    ceiling = study["evidence_budget"]
    full = render_context(blocks, titles, tokenizer, ceiling)
    graph_limit = full.tokens * study["graph75"] // ceiling
    source_limit = full.tokens - graph_limit
    candidates = graph_rows(pool["native_context"])
    model, generation = protocol["model"], protocol["options"]
    protocol_sha = file_sha256(PROTOCOL)
    for budget in range(source_limit, 0, -1):
        base = render_context(blocks, titles, tokenizer, budget)
        _, rows = pack_graph(candidates, tokenizer, full.tokens, base.text, graph_limit)
        for end in range(len(rows), 0, -1):
            selected = rows[:end]
            mixed = evidence(GRAPH, base, render_graph(selected), selected, full.tokens,
                             budget, graph_limit, tokenizer)
            n = mixed["context_tokens"]
            extra_text = render_context(blocks, titles, tokenizer, n)
            if n <= base.tokens or extra_text.tokens != n or not extra_text.text.startswith(base.text):
                continue
            plain = evidence(TEXT, extra_text, "", [], full.tokens, n, 0, tokenizer)
            pairs = []
            for variant in (plain, mixed):
                job, prompt = make_job(corpus, question, variant, model, generation, protocol_sha, pool_sha)
                pairs.append((job, variant, token_count(tokenizer, prompt)))
            if pairs[0][2] == pairs[1][2] and pairs[0][2] + generation["output_tokens"] <= generation["context_tokens"]:
                return pairs
    raise ValueError(f"no nonempty exact-length pair for {corpus.corpus_id}/{question.qid}")


def prepare(root):
    frozen_run = root / "work/run5090"
    frozen = frozen_run / "STUDY_FROZEN.json"
    study = read(frozen)
    study_sha = file_sha256(frozen)
    if study["status"] != "frozen" or study["stage"] != "confirmatory":
        raise ValueError("original frozen study required")
    stats = study["statistics"]
    family = stats["primary_family"]
    main_path = frozen_run / "main_queues" / f"main_{family}" / "queue.json"
    main = load_queue(main_path)
    if read(main_path.parent / "protocol.json")["study_sha256"] != study_sha:
        raise ValueError("main queue not bound to the frozen study")
    population = sorted((j["metadata"]["corpus_id"], j["metadata"]["question_id"])
                        for j in main["jobs"] if j["metadata"]["arm"] == "T100")
    if len(set(population)) != len(population):
        raise ValueError("duplicate question in original text arm")
    corpus_ids = {cid for cid, _ in population}
    if len(corpus_ids) != stats["primary_corpora"] or len(population) != stats["primary_questions"]:
        raise ValueError("original MuSiQue cohort differs from frozen count")
    retrieval = frozen_run / "main_retrieval"
    plan_path = retrieval / "plan.json"
    pools_path = retrieval / "pools_manifest.json"
    pools = read(pools_path)
    if pools["plan_sha256"] != file_sha256(plan_path):
        raise ValueError("retrieval plan differs from manifest")
    assembly = root / "results/k1_inputs/assembly-v2/manifest.json"
    if file_sha256(assembly) != study["inputs"]["assembly_sha256"]:
        raise ValueError("published input manifest differs from original study")
    tokenizer_path = root / "data/k1_tokenizers/qwen_tokenizer.json"
    protocol = dict(
        status="frozen", stage="development", backend="ollama", family=family,
        scope="post-main follow-up on the same MuSiQue cohort; no independent confirmation",
        model=main["model"], options=main["options"], error_policy=study["error_policy"],
        statistics=stats, seed=study["seed"], primary_metric=study["primary_metric"],
        contrast=f"{GRAPH} - {TEXT}", cohort=population,
        hypothesis="the practical benefit is supported iff lower paired corpus CI exceeds the original margin",
        pairing="same question and retrieved ranking; common exact source prefix; two full prompts and contexts match in tokens",
        selection="cap at actual original text ceiling under the frozen budget; start with about 75% source; "
                  "greedily pack intact native rows in original order within the remaining quota; "
                  "remove rows from the end until a natural text prefix has exact evidence and full-prompt parity; "
                  "decrease shared prefix budget by one only if no nonempty row subset fits; require all pairs",
        order="order by hashed question key, arm order counterbalanced by question-ID hash and alternating pairs",
        input_sha256=dict(study=study_sha, main_queue=file_sha256(main_path), pools=file_sha256(pools_path),
                          tokenizer=file_sha256(tokenizer_path), assembly=file_sha256(assembly), code=file_sha256(__file__)),
    )
    freeze_json(PROTOCOL, protocol)
    tokenizer = MemoTokenizer(Tokenizer.from_file(str(tokenizer_path)))
    records = {r["corpus_id"]: r for r in read(plan_path)["corpora"] if r["family"] == family}
    if set(records) != corpus_ids:
        raise ValueError("retrieval plan and main MuSiQue queue differ")
    by_question = {}
    for cid in sorted(corpus_ids):
        record = records[cid]
        corpus = load_public_corpus(root / record["public_path"], record["public_sha256"], tokenizer, 500)
        for question in corpus.questions:
            tokenizer.clear()
            relative = f"pools/{cid}/{config_fingerprint(question.qid)}.json"
            pool_path = retrieval / relative
            pool_sha = pools["pools"][relative]
            if file_sha256(pool_path) != pool_sha:
                raise ValueError(f"retrieval pool changed: {relative}")
            pool = read(pool_path)
            if pool["question"] != asdict(question) or pool["public_sha256"] != record["public_sha256"]:
                raise ValueError(f"pool identity differs from public data: {relative}")
            pair = matched_pair(corpus, question, pool, tokenizer, study, protocol, pool_sha)
            by_question[cid, question.qid] = (pair, relative, pool_sha)
        print(f"Prepared {len(by_question)} exact-length pairs of {len(population)}", flush=True)
    if sorted(by_question) != population:
        raise ValueError("pair population differs from original main cohort")
    entries, jobs = [], []
    order = sorted(by_question, key=lambda key: (config_fingerprint(key), key))
    first_graph = int(config_fingerprint(order[0])[-1], 16) % 2
    for index, key in enumerate(order):
        pair, relative, pool_sha = by_question[key]
        if (index + first_graph) % 2:
            pair.reverse()
        for job, variant, count in pair:
            context_path = OUT / "contexts" / f"{job['id']}.json"
            freeze_json(context_path, variant)
            entries.append(dict(id=job["id"], context_path=context_path.relative_to(OUT).as_posix(),
                                context_file_sha256=file_sha256(context_path), prompt_tokens_offline=count,
                                retrieval_pool_path=relative, retrieval_pool_sha256=pool_sha))
            jobs.append(job)
    freeze_json(CONTEXT_MANIFEST, dict(entries=entries))
    freeze_json(QUEUE, dict(format="k1-generation-queue-v1", protocol_sha256=file_sha256(PROTOCOL),
                            model=main["model"], options=main["options"], jobs=jobs))
    freeze_json(OUT / "freeze_receipt.json", dict(queue=file_sha256(QUEUE), contexts=file_sha256(CONTEXT_MANIFEST)))
    _, pairs, _ = verify_prepared()
    print(json.dumps(dict(pairs=len(pairs), contexts=describe_lengths(pairs))), flush=True)


def verify_prepared():
    receipt = read(OUT / "freeze_receipt.json")
    if receipt != dict(queue=file_sha256(QUEUE), contexts=file_sha256(CONTEXT_MANIFEST)):
        raise ValueError("frozen queue or context ledger changed")
    queue, protocol = load_queue(QUEUE), read(PROTOCOL)
    if protocol["input_sha256"]["code"] != file_sha256(__file__):
        raise ValueError("frozen preparation code changed")
    manifest = {r["id"]: r for r in read(CONTEXT_MANIFEST)["entries"]}
    pairs, paths = defaultdict(dict), {}
    if set(manifest) != {j["id"] for j in queue["jobs"]}:
        raise ValueError("context ledger and queue differ")
    for job in queue["jobs"]:
        row, meta = manifest[job["id"]], job["metadata"]
        path = OUT / row["context_path"]
        if file_sha256(path) != row["context_file_sha256"]:
            raise ValueError(f"frozen context changed: {path}")
        variant = read(path)
        if (variant["arm"] != meta["arm"] or text_sha(variant["text"]) != meta["context_sha256"]
                or variant["text"] != variant["source_text"] + variant["graph_text"]):
            raise ValueError("frozen context content differs from queue")
        key = meta["corpus_id"], meta["question_id"]
        if meta["arm"] in pairs[key]:
            raise ValueError("duplicate arm for a question")
        pairs[key][meta["arm"]] = variant, row
        paths[job["id"]] = path
    if sorted(pairs) != [tuple(key) for key in protocol["cohort"]]:
        raise ValueError("frozen queue lost original cohort members")
    for key, arms in pairs.items():
        if set(arms) != {TEXT, GRAPH}:
            raise ValueError(f"incomplete pair: {key}")
        (plain, lrow), (mixed, rrow) = arms[TEXT], arms[GRAPH]
        if (plain["context_tokens"] != mixed["context_tokens"] or plain["graph_rows"]
                or not mixed["graph_rows"] or not plain["source_text"].startswith(mixed["source_text"])
                or plain["source_text"] == mixed["source_text"]
                or lrow["prompt_tokens_offline"] != rrow["prompt_tokens_offline"]
                or lrow["retrieval_pool_sha256"] != rrow["retrieval_pool_sha256"]):
            raise ValueError(f"length, shared source, or retrieval invariant failed: {key}")
    return protocol, pairs, paths


def describe_lengths(pairs):
    variants = [arms[GRAPH][0] for arms in pairs.values()]
    def describe(values):
        return dict(mean=mean(values), minimum=min(values), maximum=max(values))
    return dict(
        evidence=describe([v["context_tokens"] for v in variants]),
        shared_source=describe([v["source_tokens"] for v in variants]),
        graph=describe([v["graph_tokens"] for v in variants]),
        graph_rows=describe([len(v["graph_rows"]) for v in variants]),
        prompt=describe([arms[GRAPH][1]["prompt_tokens_offline"] for arms in pairs.values()]),
    )


def analyze(root):
    protocol, pairs, paths = verify_prepared()
    input_manifest = root / "results/k1_inputs/assembly-v2/manifest.json"
    if file_sha256(input_manifest) != protocol["input_sha256"]["assembly"]:
        raise ValueError("evaluation references manifest changed")
    raw = read_conditions(QUEUE, OUT / "generation")
    offline = {entry["id"]: entry["prompt_tokens_offline"] for arms in pairs.values() for _, entry in arms.values()}
    if set(offline) != {row["id"] for row in raw} or any(
        row["prompt_tokens"] != offline[row["id"]] for row in raw
    ):
        raise ValueError("actual and paired offline prompt lengths must agree before opening gold")
    report, scored, corpora = evaluate(root, [(QUEUE, OUT / "generation")], OUT / "analysis",
                                      scope=protocol["scope"], contexts=paths)
    by_corpus = {(r["corpus_id"], r["arm"]): r["quality"] for r in corpora}
    ids = sorted({cid for cid, _ in pairs})
    family = protocol["family"]
    deltas = [(cid, family, by_corpus[cid, GRAPH] - by_corpus[cid, TEXT]) for cid in ids]
    stats = protocol["statistics"]
    contrast = stratified_comparison(deltas, {family: stats["primary_corpora"]},
                                     stats["bootstrap_resamples"], stats["alpha"], protocol["seed"])
    margin = stats["practical_margin"]
    contrast.update(
        practical_margin=margin, conclusion=interval_conclusion(contrast["interval"], margin),
        positive_corpora=sum(delta > 0 for _, _, delta in deltas),
        equal_corpora=sum(delta == 0 for _, _, delta in deltas),
        corpus_deltas={cid: delta for cid, _, delta in deltas},
    )
    coverage = {arm: mean(r["source_support_visible"] / r["source_support_total"] for r in scored
                          if r["arm"] == arm and r["source_support_total"]) for arm in (TEXT, GRAPH)}
    result = dict(
        scope=protocol["scope"], model=protocol["model"], questions=len(pairs), corpora=len(ids),
        responses=len(raw), terminal_statuses=dict(Counter(row["raw_status"] for row in raw)),
        equal_context_pairs=len(pairs), equal_full_prompt_pairs=len(pairs),
        lengths=describe_lengths(pairs), summary=report["summary"],
        source_support_coverage=coverage, contrast=contrast,
        sha256=dict(protocol=file_sha256(PROTOCOL), queue=file_sha256(QUEUE),
                    contexts=file_sha256(CONTEXT_MANIFEST), calls=file_sha256(OUT / "generation/calls.sqlite")),
    )
    atomic_json(OUT / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "verify", "run", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        with workflow_lock(OUT):
            prepare(args.root)
    elif args.mode == "verify":
        _, pairs, _ = verify_prepared()
        print(json.dumps(dict(pairs=len(pairs), contexts=describe_lengths(pairs))))
    elif args.mode == "run":
        sys.path.insert(0, str(args.root / "work/run5090/shared_launcher"))
        from bootstrap_shared_gpu import server5090
        with workflow_lock(OUT):
            verify_prepared()
            with server5090(OUT / "backend") as (client, _):
                print(run_scientific_queue(QUEUE, OUT / "generation", client), flush=True)
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
