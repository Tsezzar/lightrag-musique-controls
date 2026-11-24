"""One-command prospective run with checked stages, bounded retries and return ZIP."""
import argparse
import asyncio
from contextlib import contextmanager,redirect_stdout,redirect_stderr
import csv
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import zipfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor/lightrag-source'),str(ROOT/'src'),str(ROOT/'tools')]
from ragsel.k1_artifacts import freeze_json,workflow_lock
from ragsel.k1_protocol import file_sha256,config_fingerprint
from ragsel.k1_study_execution import run_scientific_queue,atomic_json,ContractError,TERMINAL
from ragsel.k1_study_policy import FAMILIES,MAIN_ARMS,BASE_SEED,chosen_subset,options,main_variants
from ragsel.k1_study_analysis import evaluate,planning_variance,figures
from ragsel.k1_study_queues import prepare_queues,prepare_repetitions
from bootstrap_5090 import server5090

WORK=ROOT/'work/run5090'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load_tool(name):
    spec=importlib.util.spec_from_file_location('study_'+name,ROOT/'tools'/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def verify_bundle():
    manifest=read(ROOT/'BUNDLE_MANIFEST.json')
    for entry in manifest['files']:
        p=(ROOT/entry['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or file_sha256(p)!=entry['sha256']:
            raise ContractError('package file changed/missing: '+entry['path'])
    return file_sha256(ROOT/'BUNDLE_MANIFEST.json')


def stage(name,function,output_directory=None):
    marker=WORK/'phases'/(name+'.json')
    if marker.exists():
        prior=read(marker)
        for relative,expected in prior.get('files',{}).items():
            if file_sha256(WORK/relative)!=expected:raise ContractError('completed stage changed: '+relative)
        print('Resume verified stage: '+name,flush=True);return prior['result']
    print('START '+name,flush=True);started=time.perf_counter()
    atomic_json(WORK/'STATUS.json',dict(status='running',stage=name,pid=os.getpid()))
    result=function()
    files={} if output_directory is None else {p.relative_to(WORK).as_posix():file_sha256(p) for p in sorted(Path(output_directory).rglob('*')) if p.is_file()}
    freeze_json(marker,dict(result=result,seconds=time.perf_counter()-started,files=files))
    print('DONE '+name,flush=True);return result


@contextmanager
def gpu_monitor():
    child=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits','-lms','1000'],
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    def sample():
        with (WORK/'gpu_samples.csv').open('a',encoding='utf-8') as out:
            for line in child.stdout:out.write(line);out.flush()
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:yield
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=15)
        thread.join(timeout=10)


def generation_pairs(queues,results):
    paths=sorted(Path(queues).glob('*/queue.json'),key=lambda p:(not p.parent.name.startswith('main_'),p.parent.name))
    return [(p,Path(results)/p.parent.name) for p in paths]


def contexts_for(pairs):
    result={}
    for queue,_ in pairs:
        for row in read(queue.parent/'contexts_manifest.json')['entries']:
            path=queue.parent/row['context_path'];expected=row.get('context_file_sha256',row.get('context_sha256'))
            if expected and file_sha256(path)!=expected:raise ContractError('context file changed')
            result[row['id']]=path
    return result


def run_groups(queues,results,client,limit=None):
    status={}
    for path,out in generation_pairs(queues,results):status[path.parent.name]=run_scientific_queue(path,out,client,limit=limit)
    return status


def retrieval(scope,corpora,out,url):
    driver=load_tool('retrieval_matrix_v2.py')
    if not (out/'plan.json').exists():driver.prepare(SimpleNamespace(scope=scope,corpora=corpora,out=out))
    for attempt in range(3):
        try:
            # Same PID as the overall workflow: Ctrl+C runs the native finally
            # blocks; a dead PID can safely resume via the outer workflow lock.
            worker=driver.Retrieval(SimpleNamespace(plan=out/'plan.json',url=url,retry_failed=attempt>0 or (out/'calls.sqlite').exists()))
            with (WORK/'logs'/(out.name+'.log')).open('a',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
                asyncio.run(worker.run())
            break
        except (ValueError,ContractError):raise
        except KeyboardInterrupt:raise
        except Exception as exc:
            with (WORK/'retrieval_retries.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(dict(scope=scope,attempt=attempt,error=f'{type(exc).__name__}: {exc}'))+'\n')
            if attempt==2:raise
    ledger=read(out/'pools_manifest.json')
    if len(ledger['pools'])!=sum(r['questions'] for r in read(out/'plan.json')['corpora']):raise ContractError('missing retrieval pools')
    return dict(pools=len(ledger['pools']),manifest_sha256=file_sha256(out/'pools_manifest.json'))


def check_smoke(out):
    from tokenizers import Tokenizer
    from ragsel.k1_public import load_public_corpus
    tok=Tokenizer.from_file(str(ROOT/'data/k1_tokenizers/qwen_tokenizer.json'));plan=read(out/'plan.json')
    record=plan['corpora'][0];corpus=load_public_corpus(ROOT/record['public_path'],record['public_sha256'],tok,500)
    flags=[]
    for q in corpus.questions:
        pool=read(out/'pools'/corpus.corpus_id/(config_fingerprint(q.qid)+'.json'))
        values={v['arm']:v for v in main_variants(corpus,q,pool,tok)}
        flags.append(values['T100']['source_text']!=values['T75']['source_text'])
    if not any(flags):raise ContractError('fresh development search still has no informative text-budget contrast')
    return dict(questions=len(flags),informative=sum(flags))


def resource_decision(dev_results,smoke,hours):
    p90={}
    for family in FAMILIES:
        con=sqlite3.connect((dev_results/family/'calls.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        try:times=[r[0] for r in con.execute("SELECT wall_seconds FROM k1_calls WHERE status IN ('complete','empty','truncated')")]
        finally:con.close()
        p90[family]=sorted(times)[math.ceil(.9*len(times))-1]
    audit=read(ROOT/'docs/k1/CONTEXT_PLAN_RESOURCE_AUDIT.json')
    smoke_cost=read(smoke/'graphs/musique-004/index_cost.json')['canonical_index_seconds']
    ratio=max(1.,smoke_cost/(60*audit['index_forecast']['musique']['historical_max_seconds_per_block']))
    index_seconds=audit['index_hours_conservative']*3600*ratio
    primary=sum(p90[f]*n*12*6 for f,n in [('cuad',16),('musique',16),('pubmedqa',11)])
    repeat=20*10*sum(p90.values())
    grid=6*12*7*sum(p90.values())
    serialization=6*12*3*sum(p90.values()) # conservative includes an extra matched reference
    decisions=dict(grid=True,serialization=True,six_k=True,repeatability=True)
    def forecast():return 1.3*(index_seconds+primary+(repeat if decisions['repeatability'] else 0)+
        (grid if decisions['grid'] else 0)+(serialization if decisions['serialization'] else 0))/3600
    # Only resource measurements enter these decisions, never answer quality.
    for item in ('serialization','grid','repeatability'):
        if forecast()>hours:decisions[item]=False
    return dict(optional=decisions,p90_seconds=p90,index_seconds_forecast=index_seconds,smoke_speed_ratio=ratio,
        total_hours_with_30_percent_reserve=forecast(),soft_hours=hours,
        mandatory_exceeds_soft_budget=1.3*(index_seconds+primary)/3600>hours,
        note='all 43 primary corpora retained; soft budget only removes predeclared auxiliary series')


def freeze_study(decision,dev_report,variance):
    audit=read(ROOT/'docs/k1/CONTEXT_PLAN_RESOURCE_AUDIT.json');records=audit['eligible_unscored_test_corpora']
    model_refs={row['role']:row['model'] for row in read(ROOT/'runtime/model_manifests/IDENTITY.json')}
    code=[*sorted((ROOT/'src/ragsel').rglob('*.py')),*sorted((ROOT/'tools').glob('*.py'))]
    contract=dict(status='frozen',stage='confirmatory',hypothesis='TG75-T100 on new MuSiQue corpora',
        primary_metric='mean paired token-F1 difference, equal weight per corpus',
        corpus_ids=[row['corpus_id'] for row in records],secondary_subset=chosen_subset(records),
        arms=list(MAIN_ARMS),models=model_refs,seed=BASE_SEED,evidence_budget=4000,source75=3000,graph75=1000,
        statistics=dict(unit='corpus',primary_family='musique',primary_corpora=16,primary_questions=192,
            bootstrap_resamples=20000,alpha=.05,secondary_source_alpha=.025,practical_margin=.02,
            leave_one_corpus_out=True,secondary_contrasts='CUAD and PubMedQA Bonferroni; other arms exploratory'),
        error_policy=dict(model_replies='retain and score all completed replies including empty and length-limited output; no quality-based retries',
            transport='at most two retries, same request, all attempts retained; missing is not zero',
            invalid_format='explicit invalid outcome under the fixed metric, not silently removed',
            extraction_truncation='processed graphs retained with native truncation metadata; failed processing cannot count as complete'),
        optional=decision['optional'],resource_forecast=decision,development_variance=variance,
        underfill='retain all questions/corpora; report T100=T75 cases, no invented padding',
        six_k_rule='remove the entire 6000-token grid before any test reply if any full request plus reserve cannot fit',
        serialization_rule='same source and graph JSON objects, common preparation before permutations; extra source-first reference if necessary',
        secondary_generator=dict(name='qwen2.5-3b-instruct-q4_k_m-gguf',digest='626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d',
            backend='llama.cpp b8967',stage='same 18-corpus subset, 648 calls on the smaller local GPU; not independent of primary'),
        inputs=dict(assembly_sha256=file_sha256(ROOT/'results/k1_inputs/assembly-v2/manifest.json'),
            development_report_sha256=file_sha256(dev_report),source_bundle_sha256=file_sha256(ROOT/'BUNDLE_MANIFEST.json')),
        code_sha256={p.relative_to(ROOT).as_posix():file_sha256(p) for p in code})
    freeze_json(WORK/'STUDY_FROZEN.json',contract)
    return dict(study_sha256=file_sha256(WORK/'STUDY_FROZEN.json'),resource_decision=decision)


def check_frozen_code():
    for relative,expected in read(WORK/'STUDY_FROZEN.json')['code_sha256'].items():
        if file_sha256(ROOT/relative)!=expected:raise ContractError('study code changed after freeze: '+relative)


def analysis(queues,results,out,scope,expected=None):
    pairs=generation_pairs(queues,results)
    report,rows,corpora=evaluate(ROOT,pairs,out,scope=scope,expected_main=expected,contexts=contexts_for(pairs))
    figures(report,corpora,out/'figures')
    if scope=='development':atomic_json(out/'variance.json',planning_variance(report))
    return dict(report_sha256=file_sha256(out/'report.json'),conditions=report['conditions'])


def prepare_secondary(main_queues,out):
    from ragsel.k1_small_backend import SMALL_MODEL
    from ragsel.k1_study_policy import make_job
    from ragsel.k1_blocks import token_count
    from tokenizers import Tokenizer
    from ragsel.k1_public import load_public_corpus
    trial=read(main_queues/'TRIALS.json');study=read(WORK/'STUDY_FROZEN.json');selected=set(study['secondary_subset'])
    assembly={r['corpus_id']:r for r in read(ROOT/'results/k1_inputs/assembly-v2/manifest.json')['corpora']}
    tok=Tokenizer.from_file(str(ROOT/'data/k1_tokenizers/qwen_tokenizer.json'));groups={f:[] for f in FAMILIES};entries={f:[] for f in FAMILIES};corpora={}
    for source_id,item in trial['trials'].items():
        meta=item['metadata'];cid=meta['corpus_id']
        if item['section']!='main' or cid not in selected or meta['arm'] not in ('V','T100','TG75'):continue
        if cid not in corpora:corpora[cid]=load_public_corpus(ROOT/assembly[cid]['public_path'],assembly[cid]['public_sha256'],tok,500)
        corpus=corpora[cid];question=next(q for q in corpus.questions if q.qid==meta['question_id']);variant=read(main_queues/item['context_file'])
        job,prompt=make_job(corpus,question,variant,SMALL_MODEL,options(corpus.family),file_sha256(WORK/'STUDY_FROZEN.json'),item['pool_sha256'])
        directory=out/corpus.family;path=directory/'contexts'/(job['id']+'.json');freeze_json(path,variant)
        entries[corpus.family].append(dict(id=job['id'],context_path=path.relative_to(directory).as_posix(),context_sha256=file_sha256(path),
            source_primary_id=source_id,prompt_tokens_offline=token_count(tok,prompt)))
        groups[corpus.family].append(job)
    for family in FAMILIES:
        directory=out/family
        freeze_json(directory/'protocol.json',dict(status='frozen',stage='confirmatory',backend='llama.cpp',
            study_sha256=file_sha256(WORK/'STUDY_FROZEN.json'),digest_kind='GGUF file SHA256',
            note='secondary model on the same 18 primary corpora, not an independent dataset',error_policy=study['error_policy']))
        freeze_json(directory/'queue.json',dict(format='k1-generation-queue-v1',protocol_sha256=file_sha256(directory/'protocol.json'),
            model=SMALL_MODEL,options=options(family),jobs=groups[family]))
        freeze_json(directory/'contexts_manifest.json',dict(entries=entries[family]))
    if sum(map(len,groups.values()))!=648:raise ContractError('secondary matrix incomplete')
    return dict(jobs=648)


def pack_results(partial=False):
    # No model weights, Python libraries or downloads in the return archive.
    target=ROOT/('RETURN_PARTIAL.zip' if partial else 'RETURN_RESULTS.zip')
    temporary=target.with_suffix('.part.zip')
    sources=[]
    for directory in (WORK,ROOT/'src',ROOT/'tools',ROOT/'config',ROOT/'data/k1_study/assembly-v2',ROOT/'results/k1_inputs/assembly-v2',ROOT/'runtime/model_manifests'):
        sources.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='RUNNING.json')
    hashes={p.relative_to(ROOT).as_posix():file_sha256(p) for p in sorted(set(sources))}
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for relative in hashes:archive.write(ROOT/relative,relative)
        archive.writestr('RETURN_MANIFEST.json',json.dumps(dict(partial=partial,files=hashes),indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:raise RuntimeError('return ZIP CRC error')
    temporary.replace(target)
    atomic_json(ROOT/'RETURN_RECEIPT.json',dict(file=target.name,sha256=file_sha256(target),bytes=target.stat().st_size,partial=partial))
    print('RETURN ARCHIVE: '+str(target),flush=True)


def execute(args):
    bundle_sha=verify_bundle();WORK.mkdir(parents=True,exist_ok=True);(WORK/'logs').mkdir(exist_ok=True)
    freeze_json(WORK/'RUN_IDENTITY.json',dict(bundle_sha256=bundle_sha,physical_root=str(ROOT.resolve()),version=1))
    # Full offline suite was run locally; local preflight remains available to the operator.
    audit=read(ROOT/'work/local_checks/public_input_audit.json')
    if not audit['verified']:raise ContractError('public input audit failed')
    with server5090(WORK/'backend',args.models_dir) as (client,gpu),gpu_monitor():
        freeze_json(WORK/'GPU_IDENTITY.json',gpu)
        development=ROOT/'work/development_primary_queues';dev_results=WORK/'development_results'
        stage('01_development_pilot',lambda:run_groups(development,dev_results,client,limit=10))
        stage('02_development_complete',lambda:run_groups(development,dev_results,client),dev_results)
        stage('03_development_analysis',lambda:analysis(development,dev_results,WORK/'development_analysis','development'),WORK/'development_analysis')
        for item in read(ROOT/'runtime/model_manifests/IDENTITY.json'):
            from ragsel.k1_inference import ModelRef
            client.unload(ModelRef(**item['model']))
        smoke=WORK/'driver_smoke'
        stage('04_fresh_native_smoke',lambda:retrieval('development',['musique-004'],smoke,client.url),smoke)
        stage('05_smoke_invariants',lambda:check_smoke(smoke))
        def freeze():
            decision=resource_decision(dev_results,smoke,args.optional_hours)
            return freeze_study(decision,WORK/'development_analysis/report.json',read(WORK/'development_analysis/variance.json'))
        stage('06_scientific_freeze',freeze)
        check_frozen_code()
        main=WORK/'main_retrieval'
        stage('07_main_retrieval',lambda:retrieval('confirmatory',['all'],main,client.url),main)
        queues=WORK/'main_queues'
        stage('08_prepare_all_test_queues',lambda:dict(main_count=prepare_queues(ROOT,main,queues,WORK/'STUDY_FROZEN.json')['main_count']),queues)
        stage('09_main_and_auxiliary_generation',lambda:run_groups(queues,WORK/'generation_results',client),WORK/'generation_results')
        repeats=WORK/'repetition_queues'
        stage('10_prepare_repetitions',lambda:dict(count=prepare_repetitions(ROOT,ROOT/'work/development_replay',repeats,WORK/'STUDY_FROZEN.json')['auxiliary_count']),repeats)
        stage('11_repeatability_generation',lambda:run_groups(repeats,WORK/'repetition_results',client),WORK/'repetition_results')
        stage('12_primary_analysis',lambda:analysis(queues,WORK/'generation_results',WORK/'final_analysis','confirmatory',3096),WORK/'final_analysis')
        stage('13_prepare_local_secondary',lambda:prepare_secondary(queues,WORK/'secondary_queues'),WORK/'secondary_queues')
        def diagnostics():
            tool=load_tool('summarize_study.py')
            return tool.summarize(ROOT,WORK)
        stage('14_cost_and_auxiliary_analysis',diagnostics,WORK/'diagnostics')
        atomic_json(WORK/'STATUS.json',dict(status='complete',main_answers=3096,secondary_prepared=648,
            optional=read(WORK/'STUDY_FROZEN.json')['optional'],next='Return RETURN_RESULTS.zip; secondary runs on the small local GPU'))
    pack_results()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--models-dir',type=Path)
    parser.add_argument('--optional-hours',type=float,default=70,help='soft planning budget; primary 43 corpora are never dropped')
    parser.add_argument('--pack-only',action='store_true');args=parser.parse_args()
    os.environ['MPLCONFIGDIR']=str(ROOT/'work/cache/matplotlib')
    WORK.mkdir(parents=True,exist_ok=True)
    with workflow_lock(WORK):
        try:
            if args.pack_only:pack_results(partial=not (WORK/'STATUS.json').exists() or read(WORK/'STATUS.json').get('status')!='complete')
            else:execute(args)
        except BaseException as exc:
            atomic_json(WORK/'ERROR.json',dict(error=f'{type(exc).__name__}: {exc}',pid=os.getpid(),time=time.time()))
            print('Stopped with all raw attempts preserved: '+str(exc),file=sys.stderr,flush=True)
            try:pack_results(partial=True)
            except Exception as packaging:print('Partial packaging failed: '+str(packaging),file=sys.stderr)
            raise


if __name__=='__main__':main()
