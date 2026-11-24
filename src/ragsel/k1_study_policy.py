"""Prospectively fixed evidence policies, paired controls and stable study keys."""
from dataclasses import asdict
import json

from .k1_baselines import answer_messages, answer_schema
from .k1_blocks import TextBlock, render_context, token_count
from .k1_context_matrix import build_matrix, graph_rows, pack_graph, render_graph, render_qwen_chat, text_sha
from .k1_protocol import config_fingerprint

FAMILIES = ('cuad', 'musique', 'pubmedqa')
MAIN_ARMS = ('V', 'BM', 'T100', 'T75', 'TG75', 'G100')
BASE_SEED = 20261110
REPEAT_SEEDS = tuple(range(20261111, 20261121))
ANSWER_LIMITS = dict(cuad=1024, musique=128, pubmedqa=32)


def rankings(pool):
    return {method: tuple(TextBlock(**row) for row in rows) for method, rows in pool['ranked_blocks'].items()}


def compact_graph(rows):
    sections = {}
    for row in rows:
        sections.setdefault((row['section'], row['label']), []).append(json.loads(row['text']))
    return '\n'.join(label + '\n' + '\n'.join(json.dumps(value, ensure_ascii=False, separators=(',', ':')) for value in values) + '\n'
                     for (_, label), values in sorted(sections.items()))


def policy(corpus, question, pool, tokenizer, budget, quota, arm):
    if type(budget) is not int or budget <= 0 or type(quota) not in (float, int) or quota not in (0, .25, .5, 1):
        raise ValueError('invalid predefined budget/quota')
    ranked = rankings(pool)
    source_limit = int(budget * (1 - quota))
    graph_limit = budget - source_limit
    source = render_context(ranked['lightrag'], {d.source_id: d.title for d in corpus.documents}, tokenizer, source_limit)
    graph, selected = pack_graph(graph_rows(pool['native_context']), tokenizer, budget, source.text, graph_limit) if quota else ('', [])
    text = source.text + graph
    count = token_count(tokenizer, text)
    if count > budget:
        raise ValueError('policy overflow')
    return dict(arm=arm, text=text, text_sha256=text_sha(text), source_text=source.text,
        source_sha256=text_sha(source.text), graph_text=graph, context_tokens=count,
        source_tokens=source.tokens, graph_tokens=token_count(tokenizer, graph), budget=budget,
        graph_quota=quota, source_limit=source_limit, graph_limit=graph_limit,
        underfill_tokens=budget-count, pieces=[asdict(piece) for piece in source.pieces], graph_rows=selected)


def main_variants(corpus, question, pool, tokenizer):
    return build_matrix(corpus, question, rankings(pool), pool['native_context'], tokenizer)


def paired_serializations(main_tg, tokenizer):
    """Prepare a common set BEFORE inference; never clip one permutation alone."""
    rows = list(main_tg['graph_rows'])
    source = main_tg['source_text']
    while True:
        native, compact = render_graph(rows), compact_graph(rows)
        texts = (source + native, native + source, source + compact)
        if all(token_count(tokenizer, text) <= main_tg['budget'] for text in texts):
            break
        if not rows:
            raise ValueError('source block alone exceeds shared permutation budget')
        rows.pop()
    reference_is_main = texts[0] == main_tg['text']
    output = []
    for arm, text in zip(('TG75' if reference_is_main else 'TG75_REFERENCE', 'ORDER_GT', 'COMPACT'), texts, strict=True):
        displayed_graph = compact if arm == 'COMPACT' else native
        row = {**main_tg, 'arm': arm, 'text': text, 'text_sha256': text_sha(text),
            'context_tokens': token_count(tokenizer, text), 'underfill_tokens': main_tg['budget'] - token_count(tokenizer, text),
            'graph_rows': rows, 'graph_facts_sha256': config_fingerprint([json.loads(r['text']) for r in rows]),
            'reference_is_main': reference_is_main, 'source_text': source,
            'graph_text': displayed_graph, 'graph_tokens': token_count(tokenizer, displayed_graph),
            'comparison_rule': 'identical source and graph objects; common preparation before any answers',
            'serialization': arm}
        output.append(row)
    return output


def chosen_subset(records):
    return [row['corpus_id'] for family in FAMILIES
            for row in sorted([r for r in records if r['family'] == family], key=lambda r: config_fingerprint(r['corpus_id']))[:6]]


def repetition_questions(corpora):
    return {(corpus.corpus_id, question.qid) for family in FAMILIES
            for corpus, question in sorted([(c, q) for c in corpora if c.family == family for q in c.questions],
                key=lambda pair: config_fingerprint([pair[0].corpus_id, pair[1].qid]))[:10]}


def options(family, seed=BASE_SEED):
    return dict(context_tokens=8192, output_tokens=ANSWER_LIMITS[family], temperature=0, seed=seed)


def make_job(corpus, question, variant, model, generation, study_sha, pool_sha):
    messages = answer_messages(corpus.family, question.text, variant['text'])
    key_data = dict(corpus_id=corpus.corpus_id, qid=question.qid, arm=variant['arm'],
        budget=variant['budget'], quota=variant['graph_quota'], source_limit=variant['source_limit'],
        context_sha256=variant['text_sha256'], model=model, options=generation, study_sha256=study_sha, pool_sha256=pool_sha)
    key = config_fingerprint(key_data)
    job = dict(id=key, metadata=dict(corpus_id=corpus.corpus_id, question_id=question.qid, arm=variant['arm'],
        family=corpus.family, budget=variant['budget'], graph_quota=variant['graph_quota'],
        context_sha256=variant['text_sha256'], seed=generation['seed']), messages=[asdict(m) for m in messages], thinking=False)
    schema = answer_schema(corpus.family)
    if schema is not None:
        job['schema'] = schema
    return job, render_qwen_chat(messages)


def planned_variants(corpus, question, pool, tokenizer, selected, grid=True, serialization=True, six_k=True):
    main = main_variants(corpus, question, pool, tokenizer)
    result = [('main', variant, None) for variant in main]
    if corpus.corpus_id not in selected:
        return result
    if grid:
        for budget in ((2000, 4000, 6000) if six_k else (2000, 4000)):
            for quota in (0, .25, .5):
                if budget == 4000 and quota in (0, .25):
                    continue
                arm = f'GRID_B{budget}_G{int(quota*100)}'
                result.append(('grid', policy(corpus, question, pool, tokenizer, budget, quota, arm), None))
    if serialization:
        tg = next(row for row in main if row['arm'] == 'TG75')
        reference, swapped, compact = paired_serializations(tg, tokenizer)
        if not reference['reference_is_main']:
            result.append(('serialization', reference, None))
        result.extend([('serialization', swapped, reference['text_sha256']),
                       ('serialization', compact, reference['text_sha256'])])
    return result
