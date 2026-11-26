"""Run the frozen MuSiQue validation experiment locally and resume from receipts."""

import argparse
import asyncio
from collections import defaultdict
import ctypes
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sqlite3
from statistics import mean
import sys
from types import SimpleNamespace
from requests.exceptions import ConnectionError, Timeout
from tokenizers import Tokenizer

from ragsel.k1_artifacts import freeze_json, workflow_lock
from ragsel.k1_blocks import TextBlock, prefix_length, render_context, token_count
from ragsel.k1_context_matrix import graph_rows, render_qwen_chat, text_sha, validate_ranked
from ragsel.k1_handoff import load_queue
from ragsel.k1_inference import CallRecorder, GenerationOptions, Message, ModelRef
from ragsel.k1_protocol import config_fingerprint, file_sha256
from ragsel.k1_public import load_public_corpus
from ragsel.k1_scoring import score_answer
from ragsel.k1_data import Reference
from ragsel.k1_statistics import interval_conclusion, stratified_comparison
from ragsel.k1_study_analysis import read_conditions
from ragsel.k1_study_execution import atomic_json, latest_rows, run_scientific_queue
from ragsel.k1_study_policy import make_job

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import RUN, read

RETRIEVAL = RUN / "retrieval"
GENERATION = RUN / "generation"
QUEUE = RUN / "answer_queue"
SUMMARIES = RUN / "summaries"
GRAPH, SUMMARY = "GRAPH_PROSE", "TEXT_SUMMARY"
END = "\n\nEnd of evidence."


def retrieval(root, url, retry_failed):
    plan = RETRIEVAL / "plan.json"
    ledger = RETRIEVAL / "pools_manifest.json"
    if ledger.exists():
        verify_pools()
        return
    sys.path.insert(0, str(root / "vendor/lightrag-source"))
    spec = importlib.util.spec_from_file_location("fresh_retrieval", root/"tools/retrieval_matrix_v2.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    packaged_input = tool.resolve_input
    def scoped_input(relative):
        if not relative.startswith("fresh/"):
            return packaged_input(relative)
        path = (RUN / relative.removeprefix("fresh/")).resolve()
        if not path.is_relative_to(RUN):
            raise ValueError("fresh retrieval input escapes the run")
        return path
    tool.resolve_input = scoped_input
    asyncio.run(tool.Retrieval(SimpleNamespace(plan=plan, url=url, retry_failed=retry_failed)).run())
    verify_pools()


def verify_pools():
    manifest = read(RETRIEVAL/"pools_manifest.json")
    if manifest["plan_sha256"] != file_sha256(RETRIEVAL/"plan.json"):
        raise ValueError("retrieval plan changed")
    if not all(file_sha256(RETRIEVAL/path)==sha for path,sha in manifest["pools"].items()):
        raise ValueError("frozen retrieval pools changed")
    return manifest["pools"]


def store_for(corpus_id):
    root = RETRIEVAL / "graphs" / corpus_id
    receipt = read(root/"index_complete.json")
    namespace = receipt["workspace"]
    objects = []
    for name in ("kv_store_entity_chunks.json", "kv_store_relation_chunks.json"):
        relative = namespace+"/"+name
        if file_sha256(root/relative) != receipt["native_files"][relative]:
            raise ValueError("native provenance store changed")
        objects.append(read(root/relative))
    return objects


def verbalize(pool, provenance):
    data = pool["native_result"]["data"]
    whitelist = {row["chunk_id"] for row in data["chunks"]}
    relations = lambda a,b:"<SEP>".join(sorted((a,b)))
    entities = {r["entity_name"]:r for r in data["entities"]}
    edges = {relations(r["src_id"],r["tgt_id"]):r for r in data["relationships"]}
    accepted, used = [], set()
    for entry in graph_rows(pool["native_context"]):
        row = json.loads(entry["text"])
        description = row["description"].replace("<SEP>", " ").strip()
        if "entity" in row:
            key = row["entity"]
            item, receipt = entities.get(key), provenance[0].get(key)
            sentence = f'“{key}” is categorized as “{row["type"]}”. {description}'
        else:
            first, second = row["entity1"], row["entity2"]
            key = relations(first,second)
            item, receipt = edges.get(key), provenance[1].get(key)
            sentence = f'“{first}” and “{second}” are related. {description}'
        if item is None or receipt is None or item.get("description") != row["description"]:
            continue
        mapping = set(receipt["chunk_ids"])
        shown = set(item["source_id"].split("<SEP>")) - {""}
        if not mapping or not shown or not mapping <= whitelist or not shown <= whitelist or sentence in used:
            continue
        used.add(sentence)
        accepted.append(dict(sentence=sentence, sources=sorted(mapping | shown), row_sha256=entry["sha256"]))
    if not accepted:
        raise ValueError("no native graph records supported by the identical source passages")
    return accepted


def sources(root):
    cohort = read(RUN/"cohort.json")
    plan = read(RETRIEVAL/"plan.json")
    tokenizer = Tokenizer.from_file(str(root/plan["tokenizer_path"]))
    pool_ledger = verify_pools()
    for record in cohort["corpora"]:
        path = RUN/record["public_path"].removeprefix("fresh/")
        corpus = load_public_corpus(path, record["public_sha256"], tokenizer, plan["block_tokens"])
        provenance = store_for(corpus.corpus_id)
        titles = {d.source_id:d.title for d in corpus.documents}
        for question in corpus.questions:
            relative = f"pools/{corpus.corpus_id}/{config_fingerprint(question.qid)}.json"
            pool = read(RETRIEVAL/relative)
            if pool["question"] != asdict(question) or pool["public_sha256"] != record["public_sha256"]:
                raise ValueError("pool is not for the frozen public question")
            ranked = tuple(TextBlock(**row) for row in pool["ranked_blocks"]["lightrag"])
            validate_ranked(corpus, ranked)
            data = pool["native_result"]["data"]["chunks"]
            if len(ranked)!=len(data) or any(x.text!=n["content"] for x,n in zip(ranked,data,strict=True)):
                raise ValueError("native and prose passage order or text differs")
            rows = verbalize(pool,provenance)
            yield corpus,question,ranked,titles,rows,pool_ledger[relative],tokenizer


def summaries(root, protocol, client):
    gemma_tokenizer = Tokenizer.from_file(str(root/"data/k1_tokenizers/gemma_tokenizer.json"))
    model = ModelRef(**protocol["models"][protocol["summary_model_role"]])
    opts = GenerationOptions(**protocol["summary_options"])
    SUMMARIES.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(SUMMARIES/"calls.sqlite") as connection:
        recorder = CallRecorder(connection)
        prior = latest_rows(recorder)
        for index,(corpus, question, ranked, titles, rows, pool_sha, _) in enumerate(sources(root),1):
            passages = "\n\n".join(titles[b.source_id]+"\n"+b.text for b in ranked)
            messages = (Message("system",protocol["summary_system"]),
                        Message("user","Research question: "+question.text+"\n\nPassages:\n"+passages))
            if sum(token_count(gemma_tokenizer,m.content) for m in messages)+opts.output_tokens+256 > opts.context_tokens:
                raise ValueError("full source evidence does not fit summarizer context")
            uid = config_fingerprint([corpus.corpus_id,question.qid,pool_sha,"summary-v1"])
            terminal = prior.get(uid)
            if terminal and terminal["status"] == "empty":
                raise ValueError(f"empty summary for {question.qid}; retained as observed")
            if not terminal or terminal["status"] not in ("complete","truncated"):
                first_attempt = 0 if not terminal else terminal["attempt"]+1
                for attempt in range(first_attempt,3):
                    try:
                        result = client.chat(model,messages,opts,recorder,uid,"summary",thinking=False)
                    except (ConnectionError, Timeout):
                        if attempt==2:
                            raise
                        continue
                    if result.status == "empty":
                        raise ValueError(f"empty summary for {question.qid}; retained as observed")
                    if result.status in ("complete","truncated"):
                        break
                terminal = latest_rows(recorder).get(uid)
            if not terminal or terminal["status"] not in ("complete","truncated"):
                raise ValueError(f"incomplete summary after retry limit for {question.qid}")
            response = json.loads(terminal["response_json"])
            prose = response["message"]["content"].strip()
            if not prose:
                raise ValueError("missing summary text")
            freeze_json(SUMMARIES/f"{uid}.json",dict(
                corpus_id=corpus.corpus_id,qid=question.qid,input_sha256=text_sha(passages),pool_sha256=pool_sha,
                input_block_ids=[b.block_id for b in ranked], text=prose, status=terminal["status"],call_id=uid,
            ))
            atomic_json(RUN/"status.json",dict(status="summarizing",complete=index,total=protocol["statistics"]["primary_questions"]))
    client.unload(model)


def matched_contexts(base, graph, summary, question, tokenizer, protocol, corpus, pool_sha):
    appendices = {GRAPH:graph,SUMMARY:summary}
    # Pre-encode whole appendices once, then use true context/prompt counts to accept a pair.
    maximum = min(protocol["appendix_cap"], *(token_count(tokenizer,text) for text in appendices.values()))
    protocol_sha = file_sha256(RUN/"experiment.json")
    for allowance in range(maximum,0,-1):
        pairs = []
        for arm, appendix in appendices.items():
            prefix = appendix[:prefix_length(tokenizer,appendix,allowance)]
            text = base.text+"\n\n"+prefix+END
            variant = dict(
                arm=arm, text=text, text_sha256=text_sha(text), context_tokens=token_count(tokenizer,text),
                budget=protocol["context_cap"], source_limit=protocol["text_cap"], graph_quota=0.,
                pieces=[asdict(p) for p in base.pieces], source_text=base.text, added_prose=prefix,
            )
            job,prompt = make_job(corpus,question,variant,protocol["models"]["generator"],
                                  protocol["answer_options"],protocol_sha,pool_sha)
            pairs.append((job,variant,token_count(tokenizer,prompt)))
        if (pairs[0][1]["context_tokens"]==pairs[1][1]["context_tokens"]<=protocol["context_cap"]
                and pairs[0][2]==pairs[1][2]
                and all(item[1]["added_prose"].strip() for item in pairs)
                and pairs[0][2]+protocol["answer_options"]["output_tokens"]<=protocol["answer_options"]["context_tokens"]):
            return pairs
    raise ValueError(f"no exact context and full-prompt match for {question.qid}")


def prepare_answers(root, protocol):
    receipt = QUEUE/"freeze_receipt.json"
    if receipt.exists():
        hashes = queue_hashes()
        if read(receipt) != hashes:
            raise ValueError("frozen answer queue differs")
        return
    jobs, entries = [], []
    pair_keys = set()
    for index,(corpus, question, ranked, titles, rows, pool_sha, tok) in enumerate(sources(root)):
        uid = config_fingerprint([corpus.corpus_id,question.qid,pool_sha,"summary-v1"])
        archive = SUMMARIES/f"{uid}.json"
        summary = read(archive)
        if summary["input_block_ids"] != [b.block_id for b in ranked] or summary["pool_sha256"]!=pool_sha:
            raise ValueError("summary and graph records must derive from the same source set")
        graph = "\n\n".join(row["sentence"] for row in rows)
        base = render_context(ranked,titles,tok,protocol["text_cap"])
        pairs = matched_contexts(base,graph,summary["text"],question,tok,protocol,corpus,pool_sha)
        if index%2:
            pairs.reverse()
        pair_keys.add((corpus.corpus_id,question.qid))
        for job,variant,count in pairs:
            path = QUEUE/"contexts"/(job["id"]+".json")
            freeze_json(path,{**variant,"shared_source_ids":summary["input_block_ids"],"summary_receipt_sha256":file_sha256(archive),
                              "eligible_graph_rows":rows if variant["arm"]==GRAPH else []})
            entries.append(dict(id=job["id"],context_path=path.relative_to(QUEUE).as_posix(),
                                context_file_sha256=file_sha256(path),prompt_tokens_offline=count))
            jobs.append(job)
    if len(jobs)!=2*len(pair_keys) or len(pair_keys)!=protocol["statistics"]["primary_questions"]:
        raise ValueError("incomplete pair population")
    protocol_file = QUEUE/"protocol.json"
    freeze_json(protocol_file,dict(status="frozen",stage="confirmatory",backend="ollama",
                                 experiment_sha256=file_sha256(RUN/"experiment.json")))
    freeze_json(QUEUE/"queue.json",dict(format="k1-generation-queue-v1",protocol_sha256=file_sha256(protocol_file),
                                      model=protocol["models"]["generator"],options=protocol["answer_options"],jobs=jobs))
    freeze_json(QUEUE/"contexts_manifest.json",dict(entries=entries))
    freeze_json(receipt,queue_hashes())


def queue_hashes():
    return {name:file_sha256(QUEUE/name) for name in ("queue.json","protocol.json","contexts_manifest.json")}


def audit_answer_queue(root, protocol):
    queue = load_queue(QUEUE/"queue.json")
    ledger = {r["id"]:r for r in read(QUEUE/"contexts_manifest.json")["entries"]}
    pairs = defaultdict(dict)
    tokenizer = Tokenizer.from_file(str(root/"data/k1_tokenizers/qwen_tokenizer.json"))
    if read(QUEUE/"freeze_receipt.json") != queue_hashes() or set(ledger)!=set(j["id"] for j in queue["jobs"]):
        raise ValueError("answer queue identity changed")
    for job in queue["jobs"]:
        row = ledger[job["id"]]
        path = QUEUE/row["context_path"]
        context = read(path)
        meta = job["metadata"]
        prompt = render_qwen_chat(tuple(Message(**m) for m in job["messages"]))
        if (file_sha256(path)!=row["context_file_sha256"] or text_sha(context["text"])!=meta["context_sha256"]
                or token_count(tokenizer,context["text"])!=context["context_tokens"]
                or token_count(tokenizer,prompt)!=row["prompt_tokens_offline"]):
            raise ValueError("queue token or evidence identity changed")
        pairs[meta["corpus_id"],meta["question_id"]][meta["arm"]]=context,row["prompt_tokens_offline"]
    for arms in pairs.values():
        if set(arms)!={GRAPH,SUMMARY}:
            raise ValueError("missing paired arm")
        (left,l_count),(right,r_count) = arms[GRAPH], arms[SUMMARY]
        if (l_count!=r_count or left["context_tokens"]!=right["context_tokens"]
                or left["source_text"]!=right["source_text"] or left["shared_source_ids"]!=right["shared_source_ids"]):
            raise ValueError("pair mismatch in length, sources or common source prefix")
    if len(pairs)!=protocol["statistics"]["primary_questions"]:
        raise ValueError("pair population changed")
    return pairs


def evaluate(root, protocol):
    pairs = audit_answer_queue(root,protocol)
    rows = read_conditions(QUEUE/"queue.json",GENERATION)
    if len(rows)!=2*len(pairs) or any(r["prompt_tokens"]!=pairs[r["corpus_id"],r["question_id"]][r["arm"]][1] for r in rows):
        raise ValueError("observed prompt counts or completeness differ")
    references = {}
    for record in read(RUN/"cohort.json")["corpora"]:
        path = RUN/record["gold_path"]
        if file_sha256(path)!=record["gold_sha256"]:
            raise ValueError("held-out references changed")
        references[record["corpus_id"]]=read(path)
    grouped = defaultdict(list)
    outcomes = []
    for row in rows:
        cid,qid,arm = row["corpus_id"],row["question_id"],row["arm"]
        item = references[cid][qid]
        reference = Reference(tuple(item["answers"]),tuple(item["support_ids"]))
        quality = score_answer("musique",qid,row["answer"],reference)["quality"] if row["answer"].strip() else 0.
        grouped[cid,arm].append(quality)
        outcomes.append({**row,"quality":quality})
    statistics = protocol["statistics"]
    corpora = sorted(references)
    by_corpus = {cid:mean(grouped[cid,GRAPH])-mean(grouped[cid,SUMMARY]) for cid in corpora}
    contrast = stratified_comparison([(cid,"musique",delta) for cid,delta in by_corpus.items()],
                                     {"musique":statistics["primary_corpora"]},
                                     statistics["bootstrap_resamples"],statistics["alpha"],protocol["seed"])
    margin = statistics["practical_margin"]
    contrast.update(practical_margin=margin,conclusion=interval_conclusion(contrast["interval"],margin),
                    corpus_deltas=by_corpus)
    lengths = [pair[GRAPH][0]["context_tokens"] for pair in pairs.values()]
    atomic_json(RUN/"result.json",dict(
        hypotheses=protocol["research_question"],questions=len(pairs),corpora=len(corpora),responses=len(rows),
        equal_context_pairs=len(pairs), equal_full_prompt_pairs=len(pairs), mean_context_tokens=mean(lengths),
        means={arm:mean(mean(grouped[cid,arm]) for cid in corpora) for arm in (GRAPH,SUMMARY)},
        contrast=contrast,protocol_sha256=file_sha256(RUN/"experiment.json"),cohort_sha256=protocol["cohort_sha256"],
    ))
    atomic_json(RUN/"question_results.json",outcomes)


def main(root, retry_failed):
    protocol = read(RUN/"experiment.json")
    if any(file_sha256(HERE/name)!=sha for name,sha in protocol["code"].items()):
        raise ValueError("frozen experiment code changed")
    if file_sha256(RUN/"cohort.json")!=protocol["cohort_sha256"]:
        raise ValueError("frozen cohort changed")
    sys.path.insert(0,str(root/"work/run5090/shared_launcher"))
    from bootstrap_shared_gpu import server5090
    with workflow_lock(RUN):
        # Prevent the idle sleep timer from suspending inference. Explicit sleep still pauses it.
        execution_state = ctypes.windll.kernel32.SetThreadExecutionState
        execution_state(0x80000001)
        try:
            atomic_json(RUN/"status.json",dict(status="starting_local_backend"))
            with server5090(RUN/"backend") as (client,gpu):
                try:
                    atomic_json(RUN/"status.json",dict(status="indexing_and_retrieval",gpu=gpu["name"]))
                    retrieval(root,client.url,retry_failed)
                    summaries(root,protocol,client)
                    prepare_answers(root,protocol)
                    audit_answer_queue(root,protocol)
                    atomic_json(RUN/"status.json",dict(status="generating_answers",total=2*protocol["statistics"]["primary_questions"]))
                    run_scientific_queue(QUEUE/"queue.json",GENERATION,client)
                    atomic_json(RUN/"status.json",dict(status="evaluating"))
                    evaluate(root,protocol)
                finally:
                    for definition in protocol["models"].values():
                        try:
                            client.unload(ModelRef(**definition))
                        except Exception as error:
                            print(f"Model unload diagnostic: {error}",file=sys.stderr)
            atomic_json(RUN/"status.json",dict(status="complete",results="result.json"))
        except Exception as exc:
            atomic_json(RUN/"status.json",dict(status="failed",error=f"{type(exc).__name__}: {exc}"))
            raise
        finally:
            execution_state(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--retry-failed",action="store_true")
    args = parser.parse_args()
    main(args.root.resolve(),args.retry_failed)
