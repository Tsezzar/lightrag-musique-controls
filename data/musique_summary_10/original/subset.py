"""Freeze and run a prefix of the fresh MuSiQue design using saved retrieval."""

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as experiment
from ragsel.k1_artifacts import freeze_json, workflow_lock
from ragsel.k1_protocol import file_sha256

ORIGINAL = experiment.RUN


def copy_checked(source, target, expected):
    if file_sha256(source) != expected:
        raise ValueError(f"source artifact has changed: {source}")
    if target.exists():
        if file_sha256(target) != expected:
            raise ValueError(f"frozen copy has changed: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    shutil.copyfile(source, temporary)
    if file_sha256(temporary) != expected:
        raise ValueError(f"source artifact changed during copying: {source}")
    temporary.replace(target)


def prepare(target, count):
    read = experiment.read
    origin_plan = ORIGINAL / "retrieval/plan.json"
    origin_protocol = ORIGINAL / "experiment.json"
    origin_cohort = ORIGINAL / "cohort.json"
    cohort = read(origin_cohort)
    plan = read(origin_plan)
    protocol = read(origin_protocol)
    if not 2 <= count < cohort["fresh_corpora"]:
        raise ValueError("a reduced corpus count of at least two is required")
    if (
        file_sha256(origin_cohort) != protocol["cohort_sha256"]
        or file_sha256(origin_protocol) != plan["confirmatory_analysis_protocol"]
        or any(file_sha256(experiment.HERE / name) != sha for name, sha in protocol["code"].items())
    ):
        raise ValueError("the original study identity changed")
    if (ORIGINAL / "generation/calls.sqlite").exists() or (ORIGINAL / "result.json").exists():
        raise ValueError("cannot freeze a pre-outcome subset of an already started answer run")
    selected = cohort["corpora"][:count]
    if [r["corpus_id"] for r in selected] != [r["corpus_id"] for r in plan["corpora"][:count]]:
        raise ValueError("the frozen corpus ordering differs between plans")
    pools = {}
    for record in selected:
        public = record["public_path"].removeprefix("fresh/")
        gold = record["gold_path"]
        copy_checked(ORIGINAL / public, target / public, record["public_sha256"])
        copy_checked(ORIGINAL / gold, target / gold, record["gold_sha256"])
        corpus_id = record["corpus_id"]
        index = Path("retrieval/graphs") / corpus_id
        receipt = read(ORIGINAL / index / "index_complete.json")
        if receipt["public_sha256"] != record["public_sha256"] or receipt["blocks"] != record["blocks"]:
            raise ValueError(f"saved index differs from cohort for {corpus_id}")
        for name, sha in {**receipt["native_files"], **{
            f"receipts/{name}": sha for name, sha in receipt["receipts"].items()
        }}.items():
            copy_checked(ORIGINAL / index / name, target / index / name, sha)
        marker = index / "index_complete.json"
        copy_checked(ORIGINAL / marker, target / marker, file_sha256(ORIGINAL / marker))
        native_pools = Path("pools") / corpus_id
        expected_questions = {q["qid"] for q in read(ORIGINAL / public)["questions"]}
        for seal in (ORIGINAL / "retrieval" / native_pools).glob("*.sha256.json"):
            relative = native_pools / seal.name.replace(".sha256.json", ".json")
            source = ORIGINAL / "retrieval" / relative
            pool = read(source)
            if pool["public_sha256"] != record["public_sha256"]:
                raise ValueError(f"retrieval data does not belong to {corpus_id}")
            expected_questions.remove(pool["question"]["qid"])
            sha = read(seal)["sha256"]
            copy_checked(source, target / "retrieval" / relative, sha)
            pools[relative.as_posix()] = sha
        if expected_questions:
            raise ValueError(f"not all selected questions have completed retrieval: {corpus_id}")
    questions = sum(r["questions"] for r in selected)
    freeze_json(target / "cohort.json", {
        **cohort, "corpora": selected, "fresh_corpora": count, "fresh_questions": questions,
        "fresh_blocks": sum(r["blocks"] for r in selected),
        "parent_cohort_sha256": file_sha256(origin_cohort),
    })
    protocol.update(
        parent_protocol_sha256=file_sha256(origin_protocol),
        sampling_deviation=(
            f"Operational restriction to the first {count} frozen corpora with finished indices and retrieval, "
            "decided after retrieval and before any summary or final answer. "
            "The reduced prospective comparison is reported separately from the original full design."
        ),
        cohort_sha256=file_sha256(target / "cohort.json"),
        statistics={**protocol["statistics"], "primary_corpora": count, "primary_questions": questions},
        code={**protocol["code"], Path(__file__).name: file_sha256(Path(__file__))},
    )
    freeze_json(target / "experiment.json", protocol)
    plan.update(
        corpora=plan["corpora"][:count],
        confirmatory_analysis_protocol=file_sha256(target / "experiment.json"),
        purpose="reuse original retrieval and index snapshots without any additional indexing",
        original_retrieval_plan_sha256=file_sha256(origin_plan),
    )
    plan["inputs"].update({
        "fresh/cohort.json": protocol["cohort_sha256"],
        "fresh/experiment.json": plan["confirmatory_analysis_protocol"],
    })
    freeze_json(target / "retrieval/plan.json", plan)
    freeze_json(target / "retrieval/pools_manifest.json", dict(
        plan_sha256=file_sha256(target / "retrieval/plan.json"), pools=pools,
        final_answers_generated=0, pool_scope=plan["pool_scope"],
    ))
    print(f"Frozen {count} corpora, {questions} completed retrievals and {2 * questions} planned answers", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpora", type=int, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    target = ORIGINAL.with_name(f"run_{args.corpora}corpora")
    with workflow_lock(target / "preparation"):
        prepare(target, args.corpora)
    if not args.prepare_only:
        experiment.RUN = target
        experiment.RETRIEVAL = target / "retrieval"
        experiment.SUMMARIES = target / "summaries"
        experiment.QUEUE = target / "answer_queue"
        experiment.GENERATION = target / "generation"
        experiment.main(args.root.resolve(), retry_failed=False)
