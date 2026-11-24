"""Compare native LightRAG records with prose expressing the same evidence."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

from tokenizers import Tokenizer

from ragsel.k1_artifacts import freeze_json, workflow_lock
from ragsel.k1_blocks import token_count
from ragsel.k1_context_matrix import render_graph, render_qwen_chat, text_sha
from ragsel.k1_handoff import load_queue
from ragsel.k1_inference import Message
from ragsel.k1_protocol import config_fingerprint, file_sha256
from ragsel.k1_statistics import interval_conclusion, stratified_comparison
from ragsel.k1_study_analysis import evaluate, read, read_conditions
from ragsel.k1_study_execution import atomic_json, run_scientific_queue

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "musique_length_control"
QUEUE = OUT / "queue.json"
PROTOCOL = OUT / "protocol.json"
CONTEXTS = OUT / "contexts_manifest.json"
NATIVE, PROSE = "NATIVE_GRAPH", "PROSE_FACTS"

ENTITY = (
    '“{name}” is categorised as “{kind}”.',
    'The category of “{name}” is “{kind}”.',
    '“{name}” belongs to the category “{kind}”.',
    '“{name}” is listed as belonging to the category “{kind}”.',
    'The material describes “{name}” as belonging to the category “{kind}”.',
)
RELATION = (
    '“{first}” and “{second}” are related.',
    'There is a connection between “{first}” and “{second}”.',
    'The material describes a connection between “{first}” and “{second}”.',
    'A relationship between “{first}” and “{second}” is described in the material.',
    'The material contains an account of the relationship between “{first}” and “{second}”.',
)
LEADS = (
    ' ', ' The description follows. ', ' The description is given below. ',
    ' The material provides the description given in the following sentences. ',
    ' The description given in the material is reproduced in full in the sentences that follow. ',
    ' A description is also given in the supplied material. Its wording is reproduced in the following sentences. ',
)
ENDINGS = ("\n", "", "\n\n", "\r\n", "\n\n\n", " ")


def prose_options(row):
    record = json.loads(row["text"])
    if set(record) == {"entity", "type", "description"}:
        heads = [pattern.format(name=record["entity"], kind=record["type"]) for pattern in ENTITY]
    elif set(record) == {"entity1", "entity2", "description"}:
        heads = [pattern.format(first=record["entity1"], second=record["entity2"]) for pattern in RELATION]
    else:
        raise ValueError("unexpected fields in native evidence record")
    if text_sha(row["text"]) != row["sha256"]:
        raise ValueError("native evidence record hash differs")
    description = record["description"].replace("<SEP>", " ")
    return [head + lead + description for lead in LEADS for head in heads]


def render_prose(rows, choices, ending):
    parts = []
    section = None
    for row, choice in zip(rows, choices, strict=True):
        separator = "\n" if row["section"] == section else "\n\n" if parts else ""
        parts.append(separator + prose_options(row)[choice])
        section = row["section"]
    return "\n\n" + "".join(parts) + ending


def prompt_count(messages, tokenizer):
    return token_count(tokenizer, render_qwen_chat(tuple(Message(**m) for m in messages)))


def with_evidence(messages, old, new):
    system, user = messages
    prefix = "Evidence:\n"
    question_marker = "\n\nQuestion: "
    old_header = prefix + old
    suffix = user["content"][len(old_header):]
    if not user["content"].startswith(old_header) or not suffix.startswith(question_marker):
        raise ValueError("original evidence or question boundary changed")
    return [system, {**user, "content": prefix + new + suffix}]


def match_prose(native, original_messages, expected_prompt, tokenizer):
    rows = sorted(native["graph_rows"], key=lambda r: (r["section"], r["rank"]))
    if not rows or render_graph(rows) != native["graph_text"]:
        raise ValueError("native record order or serialization differs")
    base, original = native["source_text"], native["text"]
    target = native["context_tokens"]
    if base + native["graph_text"] != original or token_count(tokenizer, original) != target:
        raise ValueError("source or native context changed")

    # Sum per-row token lengths to shortlist grammatical realizations.
    # Final acceptance always uses both actual full-string encodings.
    plans = {0: (0, ())}
    for row in rows:
        native_count = token_count(tokenizer, row["text"])
        counts = {}
        for index, line in enumerate(prose_options(row)):
            length = token_count(tokenizer, line + "\n")
            counts.setdefault(length, index)
        next_plans = {}
        for subtotal, (cost, previous) in plans.items():
            for length, index in counts.items():
                total = subtotal + length
                candidate = (cost + (native_count - length) ** 2, (*previous, index))
                if total not in next_plans or candidate < next_plans[total]:
                    next_plans[total] = candidate
        plans = next_plans

    # The nominal sum differs from actual context token count only at string boundaries.
    # Calibrate each possible ending with the same reference prose, without reading any answers.
    nominal, (_, reference) = next(iter(plans.items()))
    offsets = {}
    for ending in ENDINGS:
        rendered = base + render_prose(rows, reference, ending)
        offset = token_count(tokenizer, rendered) - nominal
        offsets[ending] = target - offset
    candidates = sorted(
        ((abs(length - target_sum), cost, suffix_index, choices, ending)
         for suffix_index, (ending, target_sum) in enumerate(offsets.items())
         for length, (cost, choices) in plans.items()),
    )
    for distance, _, _, choices, ending in candidates:
        prose = render_prose(rows, choices, ending)
        new = base + prose
        if token_count(tokenizer, new) != target:
            continue
        messages = with_evidence(original_messages, original, new)
        if prompt_count(messages, tokenizer) == expected_prompt:
            return new, prose, choices, ending, messages
    raise ValueError("no grammatical length match with exact context and full-prompt parity")


def prepare(root):
    original = load_queue(PRIOR / "queue.json")
    previous = read(PRIOR / "protocol.json")
    prior_manifest = read(PRIOR / "contexts_manifest.json")
    if read(PRIOR / "freeze_receipt.json") != {
            "queue": file_sha256(PRIOR / "queue.json"),
            "contexts": file_sha256(PRIOR / "contexts_manifest.json")}:
        raise ValueError("previous exact-length evidence changed")
    if file_sha256(root / "data/k1_tokenizers/qwen_tokenizer.json") != previous["input_sha256"]["tokenizer"]:
        raise ValueError("generator tokenizer changed")
    manifest = {row["id"]: row for row in prior_manifest["entries"]}
    sources = [job for job in original["jobs"] if job["metadata"]["arm"] == "GRAPH_MATCHED"]
    keys = [(job["metadata"]["corpus_id"], job["metadata"]["question_id"]) for job in sources]
    if sorted(keys) != [tuple(key) for key in previous["cohort"]]:
        raise ValueError("need the complete original paired cohort")
    protocol = dict(
        status="frozen", stage="development", backend="ollama", family=previous["family"],
        scope="post-main follow-up on the same MuSiQue cohort; representation with facts held fixed",
        cohort=previous["cohort"], model=original["model"], options=original["options"],
        statistics=previous["statistics"], seed=previous["seed"], error_policy=previous["error_policy"],
        primary_metric=previous["primary_metric"], contrast=f"{NATIVE} - {PROSE}",
        selection="preserve original common source and original native JSON exactly; verbalize every ordered "
                  "record with all names, types, endpoints and verbatim description segments; choose grammatical "
                  "templates deterministically by exact equal context and prompt lengths, preferring near-native "
                  "record lengths; no filler facts, truncation, exclusions or scoring before completion",
        hypotheses="practical advantage if lower paired CI > +2 pp; practical disadvantage if upper CI < -2 pp; "
                   "equivalent at the declared margin only if entire interval is inside [-2,+2] pp; otherwise inconclusive",
        input_sha256=dict(previous_queue=file_sha256(PRIOR / "queue.json"),
                          previous_contexts=file_sha256(PRIOR / "contexts_manifest.json"),
                          tokenizer=previous["input_sha256"]["tokenizer"], code=file_sha256(__file__),
                          assembly=previous["input_sha256"]["assembly"]),
    )
    tokenizer = Tokenizer.from_file(str(root / "data/k1_tokenizers/qwen_tokenizer.json"))
    jobs, payloads, entries = [], [], []
    # The first 192 decisions use only the evidence and tokenizer. Freeze the protocol and queue before any inference.
    protocol_sha = config_fingerprint(protocol)
    for index, job in enumerate(sources):
        entry = manifest[job["id"]]
        original_path = PRIOR / entry["context_path"]
        if file_sha256(original_path) != entry["context_file_sha256"]:
            raise ValueError("original native variant changed")
        native = read(original_path)
        n_prompt = entry["prompt_tokens_offline"]
        if prompt_count(job["messages"], tokenizer) != n_prompt:
            raise ValueError("native prompt must match original exactly")
        try:
            new, prose, choices, ending, messages = match_prose(native, job["messages"], n_prompt, tokenizer)
        except ValueError as error:
            raise ValueError(f"{keys[index]} / {len(native['graph_rows'])} records: {error}") from error
        ordered = sorted(native["graph_rows"], key=lambda row: (row["section"], row["rank"]))
        facts_sha = config_fingerprint([json.loads(row["text"]) for row in ordered])
        prose_variant = {**native, "text": new, "graph_text": prose, "text_sha256": text_sha(new),
                         "prose_template_indices": choices, "prose_ending": ending}
        pair = [(NATIVE, native, job["messages"]), (PROSE, prose_variant, messages)]
        if index % 2:
            pair.reverse()
        for arm, variant, msgs in pair:
            key = config_fingerprint([job["id"], arm, protocol_sha, variant["text_sha256"]])
            payload = {**variant, "arm": arm, "ordered_facts_sha256": facts_sha, "prior_id": job["id"]}
            payloads.append((key, payload, n_prompt, facts_sha))
            jobs.append({**job, "id": key, "messages": msgs, "metadata": {**job["metadata"], "arm": arm,
                         "context_sha256": variant["text_sha256"]}})
        if (index+1) % 24 == 0:
            print(f"Prepared {index+1}/{len(sources)} identical-fact pairs", flush=True)
    for key, payload, n_prompt, facts_sha in payloads:
        path = OUT / "contexts" / f"{key}.json"
        freeze_json(path, payload)
        entries.append(dict(id=key, context_path=path.relative_to(OUT).as_posix(),
                            context_file_sha256=file_sha256(path), prompt_tokens_offline=n_prompt,
                            ordered_facts_sha256=facts_sha))
    freeze_json(PROTOCOL, protocol)
    if file_sha256(PROTOCOL) != protocol_sha:
        raise ValueError("serialized protocol identity differs")
    freeze_json(CONTEXTS, dict(entries=entries))
    freeze_json(QUEUE, {**original, "protocol_sha256": protocol_sha, "jobs": jobs})
    freeze_json(OUT / "freeze_receipt.json", dict(queue=file_sha256(QUEUE), contexts=file_sha256(CONTEXTS)))
    _, _, pairs = verify(root)
    print(f"All {len(pairs)} pairs verified before inference.", flush=True)


def verify(root):
    if read(OUT / "freeze_receipt.json") != dict(queue=file_sha256(QUEUE), contexts=file_sha256(CONTEXTS)):
        raise ValueError("frozen preparation ledger changed")
    queue, protocol = load_queue(QUEUE), read(PROTOCOL)
    tokenizer_path = root / "data/k1_tokenizers/qwen_tokenizer.json"
    for key, path in (("code", Path(__file__)), ("tokenizer", tokenizer_path),
                      ("previous_queue", PRIOR / "queue.json"), ("previous_contexts", PRIOR / "contexts_manifest.json")):
        if protocol["input_sha256"][key] != file_sha256(path):
            raise ValueError(f"frozen input changed: {key}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    previous_queue = {j["id"]: j for j in load_queue(PRIOR / "queue.json")["jobs"]}
    previous_entries = {r["id"]: r for r in read(PRIOR / "contexts_manifest.json")["entries"]}
    entries = {r["id"]: r for r in read(CONTEXTS)["entries"]}
    if set(entries) != {j["id"] for j in queue["jobs"]}:
        raise ValueError("condition ledger changed")
    paths, pairs = {}, defaultdict(dict)
    for job in queue["jobs"]:
        entry = entries[job["id"]]
        path = OUT / entry["context_path"]
        if file_sha256(path) != entry["context_file_sha256"]:
            raise ValueError("context hash differs")
        context = read(path)
        meta = job["metadata"]
        key = meta["corpus_id"], meta["question_id"]
        arm = meta["arm"]
        if arm in pairs[key] or context["arm"] != arm or text_sha(context["text"]) != meta["context_sha256"]:
            raise ValueError("duplicate or incorrect condition")
        paths[job["id"]] = path
        pairs[key][arm] = job, context, entry
    if sorted(pairs) != [tuple(key) for key in protocol["cohort"]]:
        raise ValueError("same questions required")
    for key, pair in pairs.items():
        if set(pair) != {NATIVE, PROSE}:
            raise ValueError(f"missing condition in {key}")
        (n_job, native, n_entry), (p_job, prose, p_entry) = pair[NATIVE], pair[PROSE]
        original = previous_entries[native["prior_id"]]
        prior_job = previous_queue[native["prior_id"]]
        path = PRIOR / original["context_path"]
        if file_sha256(path) != original["context_file_sha256"]:
            raise ValueError("prior native input changed")
        expected = read(path)
        ordered = sorted(expected["graph_rows"], key=lambda row: (row["section"], row["rank"]))
        sha = config_fingerprint([json.loads(row["text"]) for row in ordered])
        generated = render_prose(ordered, prose["prose_template_indices"], prose["prose_ending"])
        if (native["text"] != expected["text"] or n_job["messages"] != prior_job["messages"]
                or any(context["source_text"] != expected["source_text"] or context["pieces"] != expected["pieces"]
                       or context["graph_rows"] != expected["graph_rows"] or context["ordered_facts_sha256"] != sha
                       or entry["ordered_facts_sha256"] != sha for _, context, entry in pair.values())
                or prose["text"] != expected["source_text"] + generated or prose["graph_text"] != generated
                or p_job["messages"] != with_evidence(n_job["messages"], native["text"], prose["text"])):
            raise ValueError(f"facts, source prefix, or instructions differ for {key}")
        if not (token_count(tokenizer, native["text"]) == token_count(tokenizer, prose["text"]) == expected["context_tokens"]
                and prompt_count(n_job["messages"], tokenizer) == prompt_count(p_job["messages"], tokenizer)
                == n_entry["prompt_tokens_offline"] == p_entry["prompt_tokens_offline"] == original["prompt_tokens_offline"]):
            raise ValueError(f"token parity failed for {key}")
    return protocol, paths, pairs


def analyze(root):
    protocol, paths, pairs = verify(root)
    if file_sha256(root / "results/k1_inputs/assembly-v2/manifest.json") != protocol["input_sha256"]["assembly"]:
        raise ValueError("reference manifest changed")
    raw = read_conditions(QUEUE, OUT / "generation")
    expected = {j["id"]: row["prompt_tokens_offline"] for pair in pairs.values() for j, _, row in pair.values()}
    if set(expected) != {r["id"] for r in raw} or any(expected[r["id"]] != r["prompt_tokens"] for r in raw):
        raise ValueError("complete actual prompt parity required before grading")
    report, _, by_corpus = evaluate(root, [(QUEUE, OUT / "generation")], OUT / "analysis",
                                     scope=protocol["scope"], contexts=paths)
    quality = {(row["corpus_id"], row["arm"]): row["quality"] for row in by_corpus}
    ids = sorted({cid for cid, _ in pairs})
    deltas = [(cid, protocol["family"], quality[cid, NATIVE] - quality[cid, PROSE]) for cid in ids]
    stats = protocol["statistics"]
    contrast = stratified_comparison(deltas, {protocol["family"]: stats["primary_corpora"]},
                                     stats["bootstrap_resamples"], stats["alpha"], protocol["seed"])
    margin = stats["practical_margin"]
    contrast.update(practical_margin=margin, conclusion=interval_conclusion(contrast["interval"], margin),
                    positive_corpora=sum(delta > 0 for _, _, delta in deltas),
                    tied_corpora=sum(delta == 0 for _, _, delta in deltas),
                    corpus_deltas={cid: delta for cid, _, delta in deltas})
    lengths = [pair[NATIVE][1]["context_tokens"] for pair in pairs.values()]
    facts = [len(pair[NATIVE][1]["graph_rows"]) for pair in pairs.values()]
    result = dict(
        scope=protocol["scope"], model=protocol["model"], questions=len(pairs), corpora=len(ids), responses=len(raw),
        equal_context_pairs=len(pairs), equal_full_prompt_pairs=len(pairs), identical_fact_pairs=len(pairs),
        evidence_tokens=dict(mean=mean(lengths), min=min(lengths), max=max(lengths)),
        records=dict(total=sum(facts), mean=mean(facts), min=min(facts), max=max(facts)),
        summary=report["summary"], contrast=contrast, terminal_statuses=dict(Counter(r["raw_status"] for r in raw)),
        sha256=dict(protocol=file_sha256(PROTOCOL), queue=file_sha256(QUEUE), contexts=file_sha256(CONTEXTS),
                    calls=file_sha256(OUT / "generation/calls.sqlite")),
    )
    atomic_json(OUT / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "verify", "run", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "verify":
        _, _, pairs = verify(args.root)
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
