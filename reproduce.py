"""Recalculate manuscript tables 5–10 from saved model outputs and references."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from ragsel.k1_data import GoldSpan, Reference
from ragsel.k1_protocol import config_fingerprint, file_sha256
from ragsel.k1_scoring import score_answer
from ragsel.k1_statistics import interval_conclusion, stratified_comparison

MAIN_ARMS = ("BM", "V", "T100", "T75", "TG75", "G100")
CONTROLS = (
    ("musique_length_control", "TEXT_MATCHED", "GRAPH_MATCHED"),
    ("musique_format_control", "PROSE_FACTS", "NATIVE_GRAPH"),
    ("musique_retrieval_control", "FLAT_RECORDS", "GRAPH_PPR"),
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(first, second):
    if not math.isclose(first, second, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"recalculated score {first} does not match archived score {second}")


def verified_data(data):
    for line in (data / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        path = (data / relative).resolve()
        if not path.is_relative_to(data) or file_sha256(path) != expected:
            raise ValueError(f"publication artifact hash mismatch: {relative}")
    return read(data / "references.json")


def read_and_score(data, name, references):
    directory = data / name
    inventory = read(directory / "inventory.json")
    rows = [json.loads(line) for line in (directory / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows) or len(rows) != inventory["cases"]:
        raise ValueError(f"duplicate or missing responses: {name}")
    archive = directory / "queue.json.gz"
    if archive.is_file():
        with gzip.open(archive, "rb") as stream:
            queue_bytes = stream.read()
        if hashlib.sha256(queue_bytes).hexdigest() != inventory["original_sha256"]["queue"]:
            raise ValueError(f"decompressed queue hash mismatch: {name}")
        queue = json.loads(queue_bytes)
        if (queue["options"] != inventory["options"] or queue["model"] != inventory["model"]
                or queue["protocol_sha256"] != file_sha256(directory / "protocol.json")):
            raise ValueError(f"queue options/model/protocol binding failed: {name}")
        jobs = {j["id"]:j for j in queue["jobs"]}
        if set(jobs) != ids:
            raise ValueError(f"queue cohort mismatch: {name}")
        ledger_path = directory / "contexts_manifest.json"
        if file_sha256(ledger_path) != inventory["original_sha256"]["contexts"]:
            raise ValueError(f"original context ledger changed: {name}")
        ledger = {row["id"]:row for row in read(ledger_path)["entries"]}
        for row in rows:
            uid = row["id"]
            if (row["messages_sha256"] != config_fingerprint(jobs[uid]["messages"])
                    or row["context_file_sha256"] != ledger[uid]["context_file_sha256"]
                    or row["prompt_tokens"] != ledger[uid]["prompt_tokens_offline"]):
                raise ValueError(f"response does not match frozen prompt: {uid}")
    for row in rows:
        if row["model_digest"] != inventory["model"]["digest"] or row["raw_status"] not in {"complete", "empty", "truncated"}:
            raise ValueError(f"unverified generation status or model: {row['id']}")
        reference = references[row["corpus_id"]][row["question_id"]]
        gold = Reference(tuple(reference["answers"]), tuple(reference["support_ids"]),
                         tuple(GoldSpan(**s) for s in reference["spans"]))
        answer = row["answer"]
        row["quality"] = score_answer(row["family"], row["question_id"], answer, gold)["quality"] if answer.strip() else 0.
        close(row["quality"], row["original_quality"])
        if row["family"] == "musique":
            support = set(gold.support_ids)
            visible = {p["source_id"] for p in row["pieces"]}
            row["support_total"] = len(support)
            row["support_visible"] = len(support & visible)
            row["recall_at_6"] = len(support & set(row["top_sources"]))/len(support)
            row["delivered_recall"] = len(support & visible)/len(support)
            row["all_supports"] = float(support <= visible)
    return rows


def averages(rows, arm, metric):
    scores = defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            scores[row["corpus_id"]].append(row[metric])
    return {corpus:mean(vals) for corpus, vals in scores.items()}


def contrast(rows, low_arm, high_arm, metric, stats, seed, alpha):
    family = rows[0]["family"]
    groups = defaultdict(dict)
    for row in rows:
        if row["arm"] in (low_arm, high_arm):
            key = row["corpus_id"], row["question_id"]
            arm = row["arm"]
            if arm in groups[key]:
                raise ValueError(f"duplicate condition for {key}")
            groups[key][arm] = row[metric]
    if any(set(arms) != {low_arm, high_arm} for arms in groups.values()):
        raise ValueError("paired questions incomplete")
    low, high = (averages(rows, arm, metric) for arm in (low_arm, high_arm))
    items = [(cid, family, high[cid]-low[cid]) for cid in sorted(low)]
    out = stratified_comparison(items, {family:len(items)}, stats["bootstrap_resamples"], alpha, seed)
    out["means"] = {low_arm:mean(low.values()), high_arm:mean(high.values())}
    out["conclusion"] = interval_conclusion(out["interval"], stats["practical_margin"])
    return out


def verify_contrast(result, original):
    close(result["estimate"], original["estimate"])
    for actual, expected in zip(result["interval"], original["interval"], strict=True):
        close(actual, expected)
    expected = original.get("conclusion", original.get("practical_conclusion"))
    if expected is not None and result["conclusion"] != expected:
        raise ValueError("practical conclusion mismatch")


def calculate(data):
    refs = verified_data(data)
    main = read(data / "main_protocol.json")
    stats, seed = main["statistics"], main["seed"]
    original = read(data / "main_reported.json")
    original_means = {(r["family"],r["arm"]):r["quality"] for r in original["summary"]}
    original_pairs = {r["family"]:r for r in original["paired"]}
    output = {f"table{n}":{} for n in range(5,11)}
    nrows = 0
    for family in ("cuad", "musique", "pubmedqa"):
        rows = read_and_score(data, "main_" + family, refs)
        nrows += len(rows)
        means = {arm:mean(averages(rows, arm, "quality").values()) for arm in MAIN_ARMS}
        for arm, value in means.items():
            close(value, original_means[(family,arm)])
        output["table5"][family] = means
        alpha = stats["alpha"] if family == stats["primary_family"] else stats["secondary_source_alpha"]
        result = contrast(rows, "T100", "TG75", "quality", stats, seed, alpha)
        verify_contrast(result, original_pairs[family])
        output["table6"][family] = result
        if family == "musique":
            output["table7"] = {
                arm:dict(mean_context_tokens=mean(r["context_tokens"] for r in rows if r["arm"] == arm),
                         visible_supports=sum(r["support_visible"] for r in rows if r["arm"] == arm),
                         total_supports=sum(r["support_total"] for r in rows if r["arm"] == arm))
                for arm in ("T75", "T100", "TG75")
            }
    if nrows != original["conditions"]:
        raise ValueError("main matrix is incomplete")
    for table_num, (name, left, right) in enumerate(CONTROLS,8):
        rows = read_and_score(data, name, refs)
        original = read(data / name / "reported_result.json")
        protocol = read(data / name / "protocol.json")
        paired = defaultdict(list)
        for row in rows:
            paired[row["corpus_id"],row["question_id"]].append((row["context_tokens"],row["prompt_tokens"]))
        if (len(paired) != original["questions"]
                or sorted(paired) != sorted(map(tuple, protocol["cohort"]))
                or any(len(pair) != 2 or len(set(pair)) != 1 for pair in paired.values())):
            raise ValueError(f"cohort or exact pair-length contract failed: {name}")
        metrics = ("quality", "recall_at_6", "delivered_recall", "all_supports") if table_num == 10 else ("quality",)
        results = {}
        for metric in metrics:
            key = "f1" if metric == "quality" else metric
            value = contrast(rows, left, right, metric, stats, seed, stats["alpha"])
            expected = original["comparisons"][key] if "comparisons" in original else original["contrast"]
            verify_contrast(value, expected)
            results[key] = value
        output[f"table{table_num}"] = dict(mean_context_tokens=mean(p[0][0] for p in paired.values()),
                                            questions=len(paired), **results)
        nrows += len(rows)
    return output, nrows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="optional destination for the recalculated JSON tables")
    args = parser.parse_args()
    calculations, count = calculate((HERE / "data").resolve())
    encoded = json.dumps(calculations, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    print(f"Verified {count:,} original responses and manuscript tables 5–10.", file=sys.stderr)
