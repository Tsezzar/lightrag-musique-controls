"""Evaluate only a complete predeclared population; corpus-level uncertainty."""
from collections import Counter,defaultdict
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import sqlite3
import numpy as np

from .k1_protocol import file_sha256,config_fingerprint
from .k1_data import Reference,GoldSpan
from .k1_blocks import DeliveredPiece
from .k1_context_packing import coverage
from .k1_scoring import score_answer
from .k1_statistics import stratified_comparison,interval_conclusion
from .k1_study_execution import TERMINAL,atomic_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def read_conditions(queue_path,results):
    queue=read(queue_path);identity=read(Path(results)/'identity.json')
    if identity!={'queue_sha256':file_sha256(queue_path)}:raise ValueError('returned queue SHA differs')
    con=sqlite3.connect((Path(results)/'calls.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        cursor=con.execute('SELECT * FROM k1_calls ORDER BY rowid');names=[item[0] for item in cursor.description]
        attempts=defaultdict(list)
        for raw in cursor:
            row=dict(zip(names,raw));attempts[row['call_id']].append(row)
    finally:con.close()
    ids={job['id'] for job in queue['jobs']}
    if set(attempts)-ids:raise ValueError('foreign raw condition')
    output=[]
    for job in queue['jobs']:
        history=attempts.get(job['id'],[]);row=history[-1] if history else None
        if row is None or row['status'] not in TERMINAL:
            raise ValueError('incomplete population; gold not opened: '+job['id'])
        if row['model_digest']!=queue['model']['digest']:raise ValueError('actual model differs from queue')
        body=json.loads(row['request_json'])
        expected=config_fingerprint(dict(model_digest=row['model_digest'],role=row['role'],body=body))
        if expected!=row['request_sha256'] or body.get('messages')!=job['messages']:raise ValueError('raw request identity mismatch')
        response=json.loads(row['response_json'])
        if 'message' in response:content=response['message'].get('content','');usage=dict(prompt_tokens=response.get('prompt_eval_count'),output_tokens=response.get('eval_count'))
        else:content=response['choices'][0]['message']['content'];usage=dict(prompt_tokens=response['usage']['prompt_tokens'],output_tokens=response['usage']['completion_tokens'])
        output.append(dict(id=job['id'],**job['metadata'],raw_status=row['status'],answer=content,model=queue['model']['name'],
            model_digest=row['model_digest'],generation_seconds=row['wall_seconds'],attempts=len(history),
            transport_attempts=sum(r['status'] not in TERMINAL for r in history),**usage))
    return output


def evaluate(root,queue_pairs,out,*,scope,expected_main=None,contexts=None,selected_subset=None):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=[]
    # Validate completeness and provenance for EVERY queue before opening gold.
    for queue_path,results in queue_pairs:raw.extend(read_conditions(Path(queue_path),Path(results)))
    if len({row['id'] for row in raw})!=len(raw):raise ValueError('duplicate evaluated condition')
    if expected_main is not None:
        main=[row for row in raw if row['arm'] in ('V','BM','T100','T75','TG75','G100') and row['seed']==20261110]
        if len(main)!=expected_main:raise ValueError('full primary population required')
    records={row['corpus_id']:row for row in read(root/'results/k1_inputs/assembly-v2/manifest.json')['corpora']}
    gold={}
    for cid in {row['corpus_id'] for row in raw}:
        record=records[cid];path=root/record['gold_path']
        if file_sha256(path)!=record['gold_sha256']:raise ValueError('reference SHA changed')
        gold[cid]=read(path)
    scored=[]
    for row in raw:
        reference=gold[row['corpus_id']][row['question_id']]
        ref=Reference(tuple(reference['answers']),tuple(reference['support_ids']),tuple(GoldSpan(**s) for s in reference['spans']))
        # A completed length-limited response is scored AS RETURNED, not retried
        # or automatically zeroed. Invalid/empty answer forms remain explicit.
        score=score_answer(row['family'],row['question_id'],row['answer'],ref,status='complete')
        if not row['answer'].strip():score={**score,'quality':0.,'valid':False,'failure':'empty_model_reply'}
        value={**row,'quality':score['quality'],'valid_answer':score['valid']}
        if contexts and row['id'] in contexts:
            context=read(contexts[row['id']])
            if context['text_sha256']!=row['context_sha256']:raise ValueError('scored evidence context differs')
            covered,total=coverage(row['family'],reference,[DeliveredPiece(**p) for p in context['pieces']])
            value.update(context_tokens=context['context_tokens'],budget=context['budget'],underfill_tokens=context['underfill_tokens'],
                source_support_visible=covered,source_support_total=total,graph_rows=len(context['graph_rows']))
        scored.append(value)
    corpus_groups=defaultdict(list)
    for row in scored:corpus_groups[(row['family'],row['corpus_id'],row['arm'],row['seed'])].append(row)
    corpus_rows=[dict(family=family,corpus_id=cid,arm=arm,seed=seed,questions=len(rows),quality=float(np.mean([r['quality'] for r in rows])),
        generation_seconds=float(np.mean([r['generation_seconds'] for r in rows])),invalid=sum(not r['valid_answer'] for r in rows),
        truncated=sum(r['raw_status']=='truncated' for r in rows)) for (family,cid,arm,seed),rows in corpus_groups.items()]
    paired=[]
    for family in ('cuad','musique','pubmedqa'):
        by={(r['corpus_id'],r['arm']):r for r in corpus_rows if r['family']==family and r['seed']==20261110}
        ids=sorted({cid for cid,arm in by if arm=='T100'} & {cid for cid,arm in by if arm=='TG75'})
        values=[(cid,family,by[cid,'TG75']['quality']-by[cid,'T100']['quality']) for cid in ids]
        if len(values)>=2:
            alpha=.05 if family=='musique' or scope=='development' else .025
            result=stratified_comparison(values,{family:len(values)},20000,alpha,20261110)
            result.update(family=family,contrast='TG75-T100',practical_margin=.02,
                practical_conclusion=interval_conclusion(result['interval'],.02),
                direction='positive' if result['interval'][0]>0 else 'negative' if result['interval'][1]<0 else 'unresolved',
                leave_one_out_estimates=[float(np.mean([x[2] for x in values if x[0]!=cid])) for cid in ids],
                corpus_deltas=[dict(corpus_id=cid,delta=value) for cid,_,value in values])
            paired.append(result)
    summary=[]
    for family,arm,seed in sorted({(row['family'],row['arm'],row['seed']) for row in corpus_rows}):
        group=[r for r in corpus_rows if (r['family'],r['arm'],r['seed'])==(family,arm,seed)]
        summary.append(dict(family=family,arm=arm,seed=seed,corpora=len(group),questions=sum(r['questions'] for r in group),
            quality=float(np.mean([r['quality'] for r in group])),invalid=sum(r['invalid'] for r in group),
            truncated=sum(r['truncated'] for r in group),mean_generation_seconds=float(np.mean([r['generation_seconds'] for r in group]))))
    for name,rows in [('question_metrics.csv',scored),('corpus_metrics.csv',corpus_rows),('summary.csv',summary)]:
        if not rows:continue
        columns=list(dict.fromkeys(key for row in rows for key in row if key!='answer'))
        with (out/name).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    report=dict(scope=scope,conditions=len(scored),corpora=len({r['corpus_id'] for r in scored}),
        terminal_statuses=dict(Counter(r['raw_status'] for r in scored)),transport_attempts=sum(r['transport_attempts'] for r in scored),
        invalid_answers=sum(not r['valid_answer'] for r in scored),summary=summary,paired=paired,
        reference_policy='published references; no judge-generated labels',
        coverage_definition='CUAD: complete annotated spans; MuSiQue/PubMedQA: any visible characters of support record, not proof of sufficient facts',
        inference_error_policy='all terminal model replies retained; transport missingness is not a zero',
        interpretation='old development only, not independent confirmation' if scope=='development' else 'fixed prospective cohort, corpus-level intervals')
    atomic_json(out/'report.json',report)
    lines=[f'# Experiment report: {scope}',f'Conditions: {len(scored)}; corpora: {report["corpora"]}.',
           f'Raw statuses: {report["terminal_statuses"]}; invalid answer forms: {report["invalid_answers"]}.',
           'Primary interpretation is restricted to this model, dataset assembly and evidence policy.']
    for result in paired:lines.append(f"- {result['family']}: TG75-T100 = {result['estimate']*100:.2f} pp; interval [{result['interval'][0]*100:.2f}, {result['interval'][1]*100:.2f}] pp; C={result['independent_corpora']}; {result['practical_conclusion']}.")
    if scope=='development':lines.append('These are development results. They do not confirm the new test hypothesis or transfer to another generator.')
    (out/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    return report,scored,corpus_rows


def planning_variance(report):
    from scipy.stats import norm,t
    mu=next((p for p in report['paired'] if p['family']=='musique'),None)
    if mu is None:return dict(available=False,reason='paired development corpora unavailable')
    values=np.asarray([r['delta'] for r in mu['corpus_deltas']],dtype=float);sd=float(np.std(values,ddof=1))
    return dict(available=sd>0,development_corpora=len(values),observed_sd=sd,
        approximate_detectable_effect_n16=(float(t.ppf(.975,15)+norm.ppf(.8))*sd/4) if sd>0 else None,
        formula='(t_.975,15 + z_.8) * s_development / sqrt(16)',
        caveat='normal/t approximation for planning, not exact bootstrap power; ten development corpora do not guarantee detection of 0.02')


def figures(report,corpus_rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    main_arms=('V','BM','T100','T75','TG75','G100')
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
    for ax,family in zip(axes,('cuad','musique','pubmedqa'),strict=True):
        rows=[r for r in report['summary'] if r['family']==family and r['seed']==20261110 and r['arm'] in main_arms]
        ax.bar([r['arm'] for r in rows],[r['quality'] for r in rows],color='0.55',edgecolor='black');ax.set_ylim(0,1)
        ax.set_title(family.upper());ax.set_ylabel('Quality (benchmark-specific)');ax.tick_params(axis='x',rotation=45)
    fig.savefig(out/'quality.png',dpi=300);fig.savefig(out/'quality.pdf');plt.close(fig)
    mu=next((r for r in report['paired'] if r['family']=='musique'),None)
    if mu:
        values=mu['corpus_deltas'];fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
        ax.bar([r['corpus_id'] for r in values],[100*r['delta'] for r in values],color='0.5',edgecolor='black')
        ax.axhline(0,color='black',linewidth=.8);ax.axhspan(-2,2,color='0.9',zorder=0)
        ax.set_ylabel('TG75 - T100, percentage points');ax.tick_params(axis='x',rotation=70)
        ax.set_title('MuSiQue: one bar per independent corpus')
        fig.savefig(out/'paired_corpora.png',dpi=300);fig.savefig(out/'paired_corpora.pdf');plt.close(fig)
