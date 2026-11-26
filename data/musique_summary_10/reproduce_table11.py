"""Verify the paired generation evidence and recalculate manuscript table 11."""
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from reproduce import close, contrast, read, verified_data, verify_contrast
from ragsel.k1_data import Reference
from ragsel.k1_protocol import config_fingerprint, file_sha256
from ragsel.k1_scoring import score_answer

GRAPH, SUMMARY = "GRAPH_PROSE", "TEXT_SUMMARY"
TERMINAL = {"complete", "empty", "truncated"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keyed(rows, key):
    items = {}
    for row in rows:
        identifier = key(row)
        require(identifier not in items, f"duplicate identifier {identifier}")
        items[identifier] = row
    return items


def read_lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def recalculate():
    references = verified_data(HERE)
    cohort, protocol, reported = (read(HERE / name) for name in ("cohort.json", "experiment.json", "result.json"))
    stats = protocol["statistics"]
    require(file_sha256(HERE / "cohort.json") == protocol["cohort_sha256"] == reported["cohort_sha256"]
            and file_sha256(HERE / "experiment.json") == reported["protocol_sha256"], "unbound study files")
    published_questions = keyed(read_lines(HERE / "questions.jsonl"), lambda q: (q["corpus_id"], q["qid"]))
    official_questions = {}
    for corpus in cohort["corpora"]:
        cid = corpus["corpus_id"]
        public = HERE / corpus["public_path"].removeprefix("fresh/")
        gold = HERE / corpus["gold_path"]
        require(file_sha256(public) == corpus["public_sha256"] and file_sha256(gold) == corpus["gold_sha256"],
                f"changed corpus or references {cid}")
        require(read(gold) == references[cid], f"changed combined references {cid}")
        official_questions.update({(cid, q["qid"]): {**q, "corpus_id": cid} for q in read(public)["questions"]})
    require(published_questions == official_questions, "published questions differ from frozen corpora")
    require(Counter(cid for cid, _ in official_questions) ==
            {x["corpus_id"]: x["questions"] for x in cohort["corpora"]}, "unexpected corpus membership")

    with gzip.open(HERE / "answer_queue/queue.json.gz", "rb") as stream:
        queue_bytes = stream.read()
    freeze = read(HERE / "answer_queue/freeze_receipt.json")
    require(sha256(queue_bytes).hexdigest() == freeze["queue.json"], "changed uncompressed queue")
    for name in ("protocol.json", "contexts_manifest.json"):
        require(file_sha256(HERE / "answer_queue" / name) == freeze[name], f"changed {name}")
    queue = json.loads(queue_bytes)
    require(queue["model"] == protocol["models"]["generator"] and queue["options"] == protocol["answer_options"]
            and queue["protocol_sha256"] == freeze["protocol.json"]
            and read(HERE / "answer_queue/protocol.json")["experiment_sha256"] == reported["protocol_sha256"],
            "queue is not bound to the experiment")
    jobs = keyed(queue["jobs"], lambda j: j["id"])
    entries = keyed(read(HERE / "answer_queue/contexts_manifest.json")["entries"], lambda e: e["id"])
    public_answers = keyed(read_lines(HERE / "answers.jsonl"), lambda r: r["id"])
    raw_answers = keyed(read_lines(HERE / "raw_answers.jsonl.gz"), lambda r: (r["call_id"], r["attempt"]))
    latest = {}
    for uid, attempt in sorted(raw_answers):
        latest[uid] = raw_answers[uid, attempt]
    require(jobs.keys() == public_answers.keys() == entries.keys() == latest.keys()
            and len(jobs) == 2 * stats["primary_questions"], "incomplete generation cohort")

    pairs = defaultdict(dict)
    rows = []
    with ZipFile(HERE / "answer_queue/contexts.zip") as contexts:
        require(set(contexts.namelist()) == {v["context_path"] for v in entries.values()}, "context inventory differs")
        for uid, job in jobs.items():
            meta = job["metadata"]
            key = (meta["corpus_id"], meta["question_id"])
            arm = meta["arm"]
            observed = latest[uid]
            request = json.loads(observed["request_json"])
            response = json.loads(observed["response_json"])
            fingerprint = config_fingerprint(dict(model_digest=observed["model_digest"], role=observed["role"], body=request))
            require(observed["status"] in TERMINAL and observed["model_digest"] == queue["model"]["digest"]
                    and observed["metadata"] == meta and fingerprint == observed["request_sha256"]
                    and request["messages"] == job["messages"] and response == observed["response"], f"invalid raw reply {uid}")
            entry = entries[uid]
            context_bytes = contexts.read(entry["context_path"])
            context = json.loads(context_bytes)
            question = official_questions[key]["text"]
            require(sha256(context_bytes).hexdigest() == entry["context_file_sha256"]
                    and sha256(context["text"].encode("utf-8")).hexdigest() == meta["context_sha256"]
                    and context["arm"] == arm and request["messages"][-1]["content"] ==
                    f"Evidence:\n{context['text']}\n\nQuestion: {question}", f"context mismatch {uid}")
            count = response["prompt_eval_count"]
            require(count == entry["prompt_tokens_offline"] == public_answers[uid]["prompt_tokens"], f"length mismatch {uid}")
            require(arm not in pairs[key], f"duplicate arm {key}")
            pairs[key][arm] = (context["source_text"], context["shared_source_ids"], context["context_tokens"], count)
            ref = references[key[0]][key[1]]
            gold = Reference(tuple(ref["answers"]), tuple(ref["support_ids"]))
            answer = response["message"]["content"]
            require(answer == public_answers[uid]["answer"], f"answer differs {uid}")
            quality = score_answer("musique", key[1], answer, gold)["quality"] if answer.strip() else 0.
            close(quality, public_answers[uid]["quality"])
            rows.append({**meta, "quality": quality})
    require(pairs.keys() == official_questions.keys() and len(pairs) == stats["primary_questions"], "question population differs")
    for key, arms in pairs.items():
        require(set(arms) == {GRAPH, SUMMARY} and arms[GRAPH] == arms[SUMMARY], f"pair parity failure {key}")

    calculated = contrast(rows, SUMMARY, GRAPH, "quality", stats, protocol["seed"], stats["alpha"])
    verify_contrast(calculated, reported["contrast"])
    for arm, value in calculated["means"].items():
        close(value, reported["means"][arm])
    context_mean = mean(arms[GRAPH][2] for arms in pairs.values())
    close(context_mean, reported["mean_context_tokens"])
    return dict(corpora=calculated["independent_corpora"], questions=len(pairs), responses=len(rows),
                equal_context_pairs=len(pairs), equal_full_prompt_pairs=len(pairs), mean_context_tokens=context_mean,
                **calculated)


if __name__ == "__main__":
    print(json.dumps(recalculate(), indent=2, ensure_ascii=False))
