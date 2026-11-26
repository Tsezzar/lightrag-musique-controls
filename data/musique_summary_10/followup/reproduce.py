"""Recalculate the three-condition comparison and summarization times."""
import gzip
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import mean, median
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from reproduce_table11 import (TERMINAL, close, config_fingerprint, contrast,
                               file_sha256, keyed, read, read_lines, recalculate,
                               require, verify_contrast)
from ragsel.k1_data import Reference
from ragsel.k1_scoring import score_answer


def contexts_at(folder):
    with ZipFile(folder / "answer_queue/contexts.zip") as archive:
        return {
            entry["id"]: json.loads(archive.read(entry["context_path"]))
            for entry in read(folder / "answer_queue/contexts_manifest.json")["entries"]
        }


def run():
    recalculate()
    for line in (HERE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        target = (HERE / relative).resolve()
        require(target.is_relative_to(HERE) and file_sha256(target) == digest, f"changed file {relative}")
    references = read(PARENT / "references.json")
    protocol = read(HERE / "experiment.json")
    reported = read(HERE / "result.json")
    require(file_sha256(HERE / "experiment.json") == reported["experiment_sha256"], "changed protocol")
    require(all(file_sha256(PARENT / name) == protocol["original"][name]
                for name in ("cohort.json", "experiment.json", "answer_queue/freeze_receipt.json",
                             "retrieval/pools_manifest.json")), "changed original comparison")
    require(all(file_sha256(HERE / "original" / name) == digest
                for name, digest in protocol["code"].items()), "changed experiment code")

    queue_bytes = gzip.decompress((HERE / "answer_queue/queue.json.gz").read_bytes())
    require(sha256(queue_bytes).hexdigest() == read(HERE / "generation_identity.json")["queue_sha256"],
            "changed generation queue")
    queue = json.loads(queue_bytes)
    require(queue["model"] == protocol["model"] and queue["options"] == protocol["options"]
            and queue["protocol_sha256"] == file_sha256(HERE / "answer_queue/protocol.json")
            and read(HERE / "answer_queue/protocol.json")["experiment_sha256"] == reported["experiment_sha256"],
            "queue parameters differ from the protocol")
    jobs = keyed(queue["jobs"], lambda job: job["id"])
    original = keyed(read_lines(PARENT / "answers.jsonl"), lambda row: row["id"])
    published = keyed(read_lines(HERE / "answers.jsonl"), lambda row: row["id"])
    new_contexts = contexts_at(HERE)
    old_contexts = contexts_at(PARENT)
    entries = keyed(read(HERE / "answer_queue/contexts_manifest.json")["entries"], lambda item: item["id"])
    raw = keyed(read_lines(HERE / "raw_answers.jsonl.gz"), lambda row: (row["call_id"], row["attempt"]))
    latest = {uid: raw[uid, attempt] for uid, attempt in sorted(raw)}
    require(jobs.keys() == latest.keys() == entries.keys() == new_contexts.keys(), "new response inventory differs")
    require(not original.keys() & jobs.keys(), "new and reused calls overlap")
    for uid, job in jobs.items():
        call = latest[uid]
        request, response = [json.loads(call[k]) for k in ("request_json", "response_json")]
        require(call["status"] in TERMINAL and call["model_digest"] == queue["model"]["digest"]
                and call["metadata"] == job["metadata"] and call["response"] == response
                and call["request_sha256"] == config_fingerprint(dict(
                    model_digest=call["model_digest"], role=call["role"], body=request))
                and request["messages"] == job["messages"], f"invalid new response {uid}")
        require(response["prompt_eval_count"] == entries[uid]["prompt_tokens_offline"], f"length mismatch {uid}")
        require(sha256(new_contexts[uid]["text"].encode("utf-8")).hexdigest() == job["metadata"]["context_sha256"]
                and request["messages"][-1]["content"].startswith(
                    f"Evidence:\n{new_contexts[uid]['text']}\n\nQuestion: "), f"invalid new context {uid}")

    groups = keyed(read(HERE / "pairs.json"), lambda row: (row["corpus_id"], row["qid"]))
    arms = {arm for pair in protocol["contrasts"] for arm in pair}
    rows, reused = [], set()
    for (cid, qid), group in groups.items():
        require(group["jobs"].keys() == arms, f"invalid arms for {qid}")
        parity = set()
        for arm, uid in group["jobs"].items():
            if arm in group["reused"]:
                row, context = original[uid], old_contexts[uid]
                answer, tokens = row["answer"], row["prompt_tokens"]
                reused.add(uid)
            else:
                context = new_contexts[uid]
                row, response = jobs[uid]["metadata"], latest[uid]["response"]
                answer, tokens = response["message"]["content"], response["prompt_eval_count"]
            shown = published[uid]
            require((cid, qid, arm) == (row["corpus_id"], row["question_id"], row["arm"])
                    == (shown["corpus_id"], shown["question_id"], shown["arm"])
                    and context["arm"] == arm and shown["reused"] == (uid in reused)
                    and (answer, tokens, context["context_tokens"]) ==
                    (shown["answer"], group["prompt_tokens"], group["context_tokens"])
                    and shown["prompt_tokens"] == tokens and shown["context_tokens"] == context["context_tokens"],
                    f"unexpected response identity {uid}")
            parity.add((context["source_text"], tuple(context["shared_source_ids"])))
            ref = references[cid][qid]
            gold = Reference(tuple(ref["answers"]), tuple(ref["support_ids"]))
            value = score_answer("musique", qid, answer, gold)["quality"]
            close(value, shown["quality"])
            rows.append(dict(id=uid, corpus_id=cid, question_id=qid, arm=arm, family="musique", quality=value))
        require(len(parity) == 1, f"shared prefix or sources differ for {qid}")
    stats = protocol["statistics"]
    require(len(groups) == stats["primary_questions"] == reported["questions"] == reported["exact_matches"]
            and len(rows) == len(published) == reported["responses"] and len(reused) == reported["reused"]
            and {row["id"] for row in rows} == published.keys() == (reused | jobs.keys())
            and len(jobs) == reported["new"], "comparison population differs from archive")
    close(mean(group["context_tokens"] for group in groups.values()), reported["mean_context_tokens"])
    comparisons = {}
    for a, b in protocol["contrasts"]:
        label = f"{a}_minus_{b}"
        saved = reported["contrasts"][label]
        nominal = contrast(rows, b, a, "quality", stats, protocol["seed"], stats["alpha"])
        adjusted = contrast(rows, b, a, "quality", stats, protocol["seed"], stats["alpha"] / len(protocol["contrasts"]))
        verify_contrast(nominal, saved)
        for x, y in zip(adjusted["interval"], saved["adjusted_interval"], strict=True):
            close(x, y)
        require(adjusted["conclusion"] == saved["adjusted_conclusion"], "changed adjusted conclusion")
        for arm, value in nominal["means"].items():
            close(value, reported["means"][arm])
        comparisons[label] = dict(estimate=nominal["estimate"], interval=nominal["interval"],
                                  adjusted_interval=adjusted["interval"], adjusted_conclusion=adjusted["conclusion"])

    calls = keyed(read_lines(HERE / "summary_timings.jsonl"), lambda row: row["call_id"])
    with ZipFile(PARENT / "summaries.zip") as archive:
        summaries = {row["call_id"]: row for name in archive.namelist()
                     if (row := json.loads(archive.read(name)))}
    require(calls.keys() == summaries.keys() and len(calls) == len(groups), "timing population differs")
    model = read(PARENT / "experiment.json")["models"]["extractor"]
    times = []
    for uid, call in calls.items():
        row = summaries[uid]
        require((call["corpus_id"], call["question_id"], call["status"]) == (row["corpus_id"], row["qid"], row["status"])
                and call["model"] == model["name"] and call["model_digest"] == model["digest"]
                and isfinite(call["wall_seconds"]) and call["wall_seconds"] > 0, f"invalid timing {uid}")
        times.append(call["wall_seconds"])
    return dict(questions=len(groups), responses=len(rows), means=reported["means"], contrasts=comparisons,
                summary_wall_seconds=dict(calls=len(times), mean=mean(times), median=median(times), total=sum(times)))


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
