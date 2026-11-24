"""Prepare the entire prospective matrix from saved pools before reading test gold."""
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path

from .k1_artifacts import freeze_json
from .k1_protocol import config_fingerprint, file_sha256
from .k1_public import load_public_corpus
from .k1_blocks import token_count
from .k1_study_policy import (FAMILIES, BASE_SEED, REPEAT_SEEDS, chosen_subset,
    planned_variants, main_variants, make_job, options, repetition_questions)
from .k1_handoff import load_queue


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


class MemoTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer=tokenizer
        self.cached=lru_cache(maxsize=1024)(lambda text,special:self.tokenizer.encode(text,add_special_tokens=special))
    def encode(self,text,add_special_tokens=False):return self.cached(text,add_special_tokens)
    def clear(self):self.cached.cache_clear()


def prepare_queues(root, retrieval_run, out, study_path):
    from tokenizers import Tokenizer
    root,retrieval_run,out=Path(root),Path(retrieval_run),Path(out)
    study=read(study_path)
    if study.get('status')!='frozen' or study.get('stage')!='confirmatory':raise ValueError('frozen study contract required')
    plan=read(retrieval_run/'plan.json');pools=read(retrieval_run/'pools_manifest.json')
    if pools['plan_sha256']!=file_sha256(retrieval_run/'plan.json'):raise ValueError('retrieval identity differs')
    if set(study['corpus_ids'])!={r['corpus_id'] for r in plan['corpora']}:raise ValueError('prospective population differs')
    if study['models']!=plan['models'] or study['seed']!=plan['seed']:raise ValueError('model/seed contract differs')
    if len(pools['pools'])!=516:raise ValueError('all 516 prospective pools required')
    tok=MemoTokenizer(Tokenizer.from_file(str(root/'data/k1_tokenizers/qwen_tokenizer.json')))
    selected=set(study['secondary_subset']);study_sha=file_sha256(study_path);model=study['models']['generator']
    groups=defaultdict(list);entries=defaultdict(list);registry={};overflows=[];comparisons=[]
    all_candidates=[]
    for record in plan['corpora']:
        corpus=load_public_corpus(root/record['public_path'],record['public_sha256'],tok,500)
        for question in corpus.questions:
            tok.clear()
            relative=f'pools/{corpus.corpus_id}/{config_fingerprint(question.qid)}.json'
            path=retrieval_run/relative
            if relative not in pools['pools'] or file_sha256(path)!=pools['pools'][relative]:raise ValueError('pool hash differs')
            pool=read(path)
            if pool['question']!=asdict(question) or pool['public_sha256']!=record['public_sha256']:raise ValueError('pool/source mismatch')
            candidates=planned_variants(corpus,question,pool,tok,selected,
                grid=study['optional']['grid'],serialization=study['optional']['serialization'],six_k=study['optional']['six_k'])
            for section,variant,reference_text_sha in candidates:
                generation=options(corpus.family,study['seed'])
                job,prompt=make_job(corpus,question,variant,model,generation,study_sha,pools['pools'][relative])
                n=token_count(tok,prompt)
                if n+generation['output_tokens']>generation['context_tokens']:
                    if section=='main':raise ValueError('main request overflow; no clipping or selective exclusion allowed')
                    overflows.append(dict(id=job['id'],section=section,budget=variant['budget'],corpus_id=corpus.corpus_id,qid=question.qid))
                all_candidates.append((section,corpus.family,job,variant,n,relative,pools['pools'][relative],reference_text_sha))
    # Predeclared all-or-none rule: a 6000-token grid that cannot fit every
    # question is removed as a whole before the first test answer, never clipped.
    drop_6k=any(row['budget']==6000 for row in overflows)
    if any(row['budget']!=6000 for row in overflows):raise ValueError('non-6000 auxiliary request overflow')
    removed=0
    for section,family,job,variant,n,relative,pool_sha,reference in all_candidates:
        if drop_6k and section=='grid' and variant['budget']==6000:removed+=1;continue
        group=f'{section}_{family}'
        if job['id'] in registry:
            if registry[job['id']]['metadata']!=job['metadata']:raise ValueError('condition collision')
            continue
        context_path=out/group/'contexts'/(job['id']+'.json')
        freeze_json(context_path,variant)
        entry=dict(id=job['id'],context_path=context_path.relative_to(out/group).as_posix(),
            context_file_sha256=file_sha256(context_path),prompt_tokens_offline=n,
            retrieval_pool_path=relative,retrieval_pool_sha256=pool_sha,reference_text_sha256=reference)
        groups[group].append(job);entries[group].append(entry)
        registry[job['id']]=dict(section=section,queue=group,metadata=job['metadata'],context_file=f'{group}/{entry["context_path"]}',
            context_file_sha256=entry['context_file_sha256'],pool_path=relative,pool_sha256=pool_sha,
            prompt_tokens_offline=n,reference_text_sha256=reference)
    for group,jobs in groups.items():
        family=group.rsplit('_',1)[1]
        # Deterministic counterbalancing before inference, preserving arm labels.
        jobs.sort(key=lambda job:(config_fingerprint([job['metadata']['corpus_id'],job['metadata']['question_id']]),
                                   (list(('V','BM','T100','T75','TG75','G100')).index(job['metadata']['arm'])-
                                    int(config_fingerprint(job['metadata']['question_id'])[:8],16))%6
                                   if job['metadata']['arm'] in ('V','BM','T100','T75','TG75','G100') else job['metadata']['arm']))
        protocol=dict(status='frozen',stage='confirmatory',backend='ollama',study_sha256=study_sha,
            family=family,section=group.rsplit('_',1)[0],error_policy=study['error_policy'],
            six_k_dropped_for_full_window=drop_6k,full_prompt_counts='exact supported Qwen template, checked against runtime on every reply')
        freeze_json(out/group/'protocol.json',protocol)
        freeze_json(out/group/'queue.json',dict(format='k1-generation-queue-v1',protocol_sha256=file_sha256(out/group/'protocol.json'),
            model=model,options=options(family,study['seed']),jobs=jobs))
        freeze_json(out/group/'contexts_manifest.json',dict(entries=entries[group]))
        load_queue(out/group/'queue.json')
    main_count=sum(value['section']=='main' for value in registry.values())
    if main_count!=3096:raise ValueError('main matrix incomplete')
    freeze_json(out/'TRIALS.json',dict(study_sha256=study_sha,pools_manifest_sha256=file_sha256(retrieval_run/'pools_manifest.json'),
        trials=registry,queues={group:f'{group}/queue.json' for group in groups},main_count=main_count,
        auxiliary_count=len(registry)-main_count,window_exclusions=overflows,six_k_removed_calls=removed))
    return read(out/'TRIALS.json')


def prepare_repetitions(root,development_pools,out,study_path):
    from tokenizers import Tokenizer
    root,development_pools,out=Path(root),Path(development_pools),Path(out)
    study=read(study_path);study_sha=file_sha256(study_path);assembly=read(root/'results/k1_inputs/assembly-v2/manifest.json')
    tok=MemoTokenizer(Tokenizer.from_file(str(root/'data/k1_tokenizers/qwen_tokenizer.json')))
    corpora=[load_public_corpus(root/record['public_path'],record['public_sha256'],tok,500)
             for record in assembly['corpora'] if record['split'] in ('train','calibration')]
    chosen=repetition_questions(corpora);ledger=read(development_pools/'pools_manifest.json')['pools']
    registry={};groups=defaultdict(list);entries=defaultdict(list)
    if study['optional']['repeatability']:
        for corpus in corpora:
            for question in corpus.questions:
                if (corpus.corpus_id,question.qid) not in chosen:continue
                tok.clear();relative=f'pools/{corpus.corpus_id}/{config_fingerprint(question.qid)}.json';path=development_pools/relative
                if file_sha256(path)!=ledger[relative]:raise ValueError('development pool changed')
                pool=read(path);variants=[v for v in main_variants(corpus,question,pool,tok) if v['arm'] in ('T100','TG75')]
                for seed in REPEAT_SEEDS:
                    group=f'repeat_s{seed}_{corpus.family}'
                    for variant in variants:
                        job,prompt=make_job(corpus,question,variant,study['models']['generator'],options(corpus.family,seed),study_sha,ledger[relative])
                        path=out/group/'contexts'/(job['id']+'.json');freeze_json(path,variant)
                        entry=dict(id=job['id'],context_path=path.relative_to(out/group).as_posix(),context_file_sha256=file_sha256(path),
                                   prompt_tokens_offline=token_count(tok,prompt))
                        groups[group].append(job);entries[group].append(entry)
                        registry[job['id']]=dict(section='repeatability',queue=group,metadata=job['metadata'],
                            context_file=f'{group}/{entry["context_path"]}',context_file_sha256=entry['context_file_sha256'])
    for group,jobs in groups.items():
        family=group.rsplit('_',1)[1];seed=jobs[0]['metadata']['seed']
        freeze_json(out/group/'protocol.json',dict(status='frozen',stage='development',backend='ollama',
            study_sha256=study_sha,purpose='predeclared repeatability; seeds are not independent corpora',seed=seed))
        freeze_json(out/group/'queue.json',dict(format='k1-generation-queue-v1',protocol_sha256=file_sha256(out/group/'protocol.json'),
            model=study['models']['generator'],options=options(family,seed),jobs=jobs))
        freeze_json(out/group/'contexts_manifest.json',dict(entries=entries[group]));load_queue(out/group/'queue.json')
    if study['optional']['repeatability'] and len(registry)!=600:raise ValueError('repeatability population incomplete')
    freeze_json(out/'TRIALS.json',dict(trials=registry,queues={group:f'{group}/queue.json' for group in groups},
        main_count=0,auxiliary_count=len(registry),stage='development_repetitions'))
    return read(out/'TRIALS.json')
