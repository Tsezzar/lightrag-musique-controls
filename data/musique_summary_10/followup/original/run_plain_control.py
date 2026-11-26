"""Compare original paragraphs with the two saved arms on fresh MuSiQue."""
import argparse
from collections import defaultdict
import ctypes
from itertools import product
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as paired
from ragsel.k1_artifacts import freeze_json, workflow_lock
from ragsel.k1_data import Reference
from ragsel.k1_inference import ModelRef
from ragsel.k1_protocol import file_sha256
from ragsel.k1_scoring import score_answer
from ragsel.k1_statistics import interval_conclusion, stratified_comparison
from ragsel.k1_study_analysis import read_conditions
from ragsel.k1_study_execution import atomic_json, run_scientific_queue

ORIGINAL = paired.RUN.with_name("run_10corpora")
CONTROL = ORIGINAL.with_name("run_10corpora_plain_control")
QUEUE = CONTROL / "answer_queue"
RESULTS = CONTROL / "generation"
TEXT = "RAW_CONTINUATION"
GRAPH, SUMMARY = paired.GRAPH, paired.SUMMARY
ARMS = (TEXT, GRAPH, SUMMARY)
CONTRASTS = ((SUMMARY, TEXT), (GRAPH, TEXT), (SUMMARY, GRAPH))
CUT_RADIUS = 48
MAX_REDUCTION = 64
read = paired.read


def protocol_for(parent):
    inputs = (
        "experiment.json", "cohort.json", "retrieval/pools_manifest.json",
        "answer_queue/freeze_receipt.json", "generation/calls.sqlite", "generation/identity.json",
    )
    return dict(
        status="frozen", stage="confirmatory", backend="ollama",
        interpretation="Follow-up on previously evaluated questions, not an independent replication.",
        primary_contrast="TEXT_SUMMARY minus RAW_CONTINUATION",
        contrasts=[list(contrast) for contrast in CONTRASTS],
        intervals="95% nominal and Bonferroni-adjusted intervals for the three pairwise contrasts",
        statistics=parent["statistics"], seed=parent["seed"], model=parent["models"]["generator"],
        options=parent["answer_options"],
        original={name: file_sha256(ORIGINAL / name) for name in inputs},
        code={name: file_sha256(CONTROL.parent / name) for name in (Path(__file__).name, "run.py")},
        matching=dict(
            order="Keep the original prefix. Try a contiguous continuation, then a paragraph break. "
                  "Take exact original text from the same ranked passage list. Search cut points nearest the token boundary.",
            exception="If no match at the old length, start at the available original-text limit and shorten the three "
                      "contexts together. Regenerate both original arms at that question's shorter matched length.",
            cut_radius_characters=CUT_RADIUS, maximum_additional_reduction_tokens=MAX_REDUCTION,
        ),
    )


def candidates(tok, prefix, text, separator, target):
    body = prefix + separator + text
    middle = paired.prefix_length(tok, body, target - paired.token_count(tok, paired.END))
    start = len(prefix) + len(separator) + 1
    indices = range(max(start, middle - CUT_RADIUS), min(len(body), middle + CUT_RADIUS) + 1)
    for index in sorted(indices, key=lambda n: (abs(n - middle), n)):
        context = body[:index] + paired.END
        tail = body[start - 1:index]
        if tail.strip() and paired.token_count(tok, context) == target:
            yield context, tail


def choose(root, parent, study_sha):
    paired.RUN = ORIGINAL
    paired.RETRIEVAL = ORIGINAL / "retrieval"
    paired.QUEUE = ORIGINAL / "answer_queue"
    saved = paired.audit_answer_queue(root, parent)
    original_jobs = {
        (job["metadata"]["corpus_id"], job["metadata"]["question_id"], job["metadata"]["arm"]): job["id"]
        for job in read(paired.QUEUE / "queue.json")["jobs"]
    }
    for corpus, question, ranked, titles, _graph, pool_sha, tok in paired.sources(root):
        key = corpus.corpus_id, question.qid
        arms = saved[key]
        old_context, old_tokens = arms[GRAPH]
        prefix = old_context["source_text"]
        block_ids = [block.block_id for block in ranked]
        full = "".join(f"[{b.block_id}] {titles[b.source_id]}\n{b.text}\n\n" for b in ranked)
        if not full.startswith(prefix) or block_ids != old_context["shared_source_ids"]:
            raise ValueError(f"stored prefix or ranked sources changed: {key}")
        remaining = full[len(prefix):]
        old_budget = old_context["context_tokens"]

        def job_candidates(arm, tail, separators, target):
            for sep in separators:
                for text, addition in candidates(tok, prefix, tail, sep, target):
                    variant = dict(arm=arm, text=text, text_sha256=paired.text_sha(text), budget=parent["context_cap"],
                                   graph_quota=0., source_limit=parent["text_cap"], context_tokens=target,
                                   source_text=prefix, shared_source_ids=block_ids, added_prose=addition, separator=sep)
                    job, prompt = paired.make_job(corpus, question, variant, parent["models"]["generator"],
                                                  parent["answer_options"], study_sha, pool_sha)
                    yield paired.token_count(tok, prompt), (job, variant)

        raw = ((tokens, pair) for tokens, pair in job_candidates(TEXT, remaining, ("", "\n\n"), old_budget)
               if tokens == old_tokens)
        choice = next(raw, None)
        if choice:
            rows = {TEXT: choice[1]}
            old_ids = {arm: original_jobs[*key, arm] for arm in (GRAPH, SUMMARY)}
            yield key, rows, old_ids, old_budget, old_tokens
            continue

        upper = min(old_budget - 1, paired.token_count(tok, full + paired.END))
        for target in range(upper, upper - MAX_REDUCTION - 1, -1):
            raw_options = dict(job_candidates(TEXT, remaining, ("", "\n\n"), target))
            if not raw_options:
                continue
            others = {arm: dict(job_candidates(arm, arms[arm][0]["added_prose"], ("\n\n",), target))
                      for arm in (GRAPH, SUMMARY)}
            for tokens, raw_pair in raw_options.items():
                if all(tokens in arm_options for arm_options in others.values()):
                    rows = {TEXT: raw_pair, **{arm: options[tokens] for arm, options in others.items()}}
                    yield key, rows, {}, target, tokens
                    break
            else:
                continue
            break
        else:
            raise ValueError(f"no equal source-text context for {key}")


def prepare(root, parent):
    protocol = protocol_for(parent)
    marker = CONTROL / "experiment.json"
    freeze_json(marker, protocol)
    study_sha = file_sha256(marker)
    jobs, entries, groups = [], [], []
    for index, (key, rows, old_ids, context_tokens, prompt_tokens) in enumerate(choose(root, parent, study_sha)):
        identifiers = dict(old_ids)
        order = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
        for arm in order:
            if arm not in rows:
                continue
            job, variant = rows[arm]
            uid = job["id"]
            path = QUEUE / "contexts" / (uid + ".json")
            freeze_json(path, variant)
            entries.append(dict(id=uid, context_path=path.relative_to(QUEUE).as_posix(),
                                context_file_sha256=file_sha256(path), prompt_tokens_offline=prompt_tokens))
            jobs.append(job)
            identifiers[arm] = uid
        groups.append(dict(corpus_id=key[0], qid=key[1], jobs=identifiers, reused=list(old_ids),
                           context_tokens=context_tokens, prompt_tokens=prompt_tokens))
    if len(groups) != protocol["statistics"]["primary_questions"]:
        raise ValueError("incomplete follow-up cohort")
    plan = QUEUE / "protocol.json"
    freeze_json(plan, dict(status="frozen", stage=protocol["stage"], backend=protocol["backend"],
                           experiment_sha256=study_sha))
    freeze_json(QUEUE / "queue.json", dict(format="k1-generation-queue-v1", protocol_sha256=file_sha256(plan),
                                          model=protocol["model"], options=protocol["options"], jobs=jobs))
    freeze_json(QUEUE / "contexts_manifest.json", dict(entries=entries))
    freeze_json(CONTROL / "pairs.json", groups)
    print(f"Frozen {len(groups)} questions and {len(jobs)} new answers; {len(ARMS) * len(groups) - len(jobs)} reused.",
          flush=True)
    return protocol, jobs, groups


def evaluate(protocol, groups):
    old_rows = read_conditions(ORIGINAL / "answer_queue/queue.json", ORIGINAL / "generation")
    new_rows = read_conditions(QUEUE / "queue.json", RESULTS)
    originals = {row["id"]: row for row in old_rows}
    generated = {row["id"]: row for row in new_rows}
    if originals.keys() & generated.keys():
        raise ValueError("new and reused response identities overlap")
    observed, results = {**originals, **generated}, []
    for group in groups:
        for arm, uid in group["jobs"].items():
            row = observed[uid]
            if (row["corpus_id"], row["question_id"], row["arm"], row["prompt_tokens"]) != (
                group["corpus_id"], group["qid"], arm, group["prompt_tokens"]
            ) or (uid in originals) != (arm in group["reused"]):
                raise ValueError(f"unexpected identity or prompt length: {uid}")
            results.append({**row, "reused": uid in originals, "context_tokens": group["context_tokens"]})

    references = {}
    for corpus in read(ORIGINAL / "cohort.json")["corpora"]:
        path = ORIGINAL / corpus["gold_path"]
        if file_sha256(path) != corpus["gold_sha256"]:
            raise ValueError("reference file changed")
        references[corpus["corpus_id"]] = read(path)
    scores = defaultdict(list)
    for row in results:
        cid, qid, arm = row["corpus_id"], row["question_id"], row["arm"]
        gold = references[cid][qid]
        reference = Reference(tuple(gold["answers"]), tuple(gold["support_ids"]))
        row["quality"] = score_answer("musique", qid, row["answer"], reference)["quality"]
        scores[cid, arm].append(row["quality"])
    corpora = sorted(references)
    means = {(cid, arm): mean(scores[cid, arm]) for cid, arm in product(corpora, ARMS)}
    stats, contrasts = protocol["statistics"], {}
    for contender, reference in CONTRASTS:
        deltas = {cid: means[cid, contender] - means[cid, reference] for cid in corpora}
        rows = [(cid, "musique", delta) for cid, delta in deltas.items()]
        kwargs = dict(expected_source_counts={"musique": stats["primary_corpora"]},
                      resamples=stats["bootstrap_resamples"], seed=protocol["seed"])
        nominal = stratified_comparison(rows, alpha=stats["alpha"], **kwargs)
        adjusted = stratified_comparison(rows, alpha=stats["alpha"] / len(CONTRASTS), **kwargs)
        margin = stats["practical_margin"]
        contrasts[f"{contender}_minus_{reference}"] = dict(
            nominal, adjusted_interval=adjusted["interval"], adjusted_alpha=adjusted["alpha"],
            practical_margin=margin, conclusion=interval_conclusion(nominal["interval"], margin),
            adjusted_conclusion=interval_conclusion(adjusted["interval"], margin), corpus_deltas=deltas,
        )
    outcome = dict(questions=len(groups), corpora=len(corpora), responses=len(results), new=len(generated),
                   reused=len(results) - len(generated), exact_matches=len(groups),
                   mean_context_tokens=mean(group["context_tokens"] for group in groups),
                   means={arm: mean(means[cid, arm] for cid in corpora) for arm in ARMS},
                   contrasts=contrasts, experiment_sha256=file_sha256(CONTROL / "experiment.json"))
    atomic_json(CONTROL / "question_results.json", results)
    atomic_json(CONTROL / "result.json", outcome)


def main(root, mode):
    parent = read(ORIGINAL / "experiment.json")
    with workflow_lock(CONTROL):
        protocol, jobs, groups = prepare(root, parent)
        if mode == "prepare":
            return
        if mode == "verify":
            evaluate(protocol, groups)
            return
        sys.path.insert(0, str(root / "work/run5090/shared_launcher"))
        from bootstrap_shared_gpu import server5090
        set_power = ctypes.windll.kernel32.SetThreadExecutionState
        set_power(0x80000001)
        try:
            atomic_json(CONTROL / "status.json", dict(status="starting_local_backend", total=len(jobs)))
            with server5090(CONTROL / "backend") as (client, gpu):
                atomic_json(CONTROL / "status.json", dict(status="generating_answers", total=len(jobs), gpu=gpu["name"]))
                try:
                    run_scientific_queue(QUEUE / "queue.json", RESULTS, client)
                finally:
                    client.unload(ModelRef(**protocol["model"]))
            evaluate(protocol, groups)
            atomic_json(CONTROL / "status.json", dict(status="complete", results="result.json"))
        except Exception as error:
            atomic_json(CONTROL / "status.json", dict(status="failed", error=f"{type(error).__name__}: {error}"))
            raise
        finally:
            set_power(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "run", "verify"), default="run")
    args = parser.parse_args()
    main(args.root.resolve(), args.mode)
