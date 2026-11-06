"""Transparent retrieval metrics and optional reference-term coverage (not an LLM judge)."""
from __future__ import annotations
import argparse
import json
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from .retrieval import Retriever, ROOT, load_corpus
from .engine import Assistant

EVAL_PATH = ROOT / 'data' / 'evaluation_set.json'


def load_cases():
    return json.loads(EVAL_PATH.read_text(encoding='utf-8'))['cases']


def relevant(source: dict, case: dict) -> bool:
    return any(rule['url_contains'] in source['url'] and re.search(rule['heading_regex'], source['heading'], re.I)
               for rule in case.get('relevant_sources', []))


def reference_coverage(answer: str, groups: list[list[str]]) -> float | None:
    if not groups:
        return None
    normalized = re.sub(r'[^a-z0-9]+', '', answer.lower())
    return sum(any(re.sub(r'[^a-z0-9]+', '', term.lower()) in normalized for term in group) for group in groups) / len(groups)


def evaluate(retriever: Retriever, cases: list[dict], *, top_k=6, generate=False,
             provider='Source excerpts', model='', api_key='', progress=None) -> dict:
    rows=[]
    assistant=Assistant(retriever)
    for i,case in enumerate(cases):
        start=time.perf_counter()
        hits=retriever.search(case['question'], k=top_k)
        sources=[h.to_dict() for h in hits]
        rank=next((n for n,s in enumerate(sources,1) if relevant(s,case)), None)
        row={'id':case['id'],'question':case['question'],'split':case['split'],
             'hit_at_k':bool(rank),'reciprocal_rank':1/rank if rank else 0.0,
             'first_relevant_rank':rank,'expected_abstain':case.get('expected_abstain',False),
             'sources':[{'url':s['url'],'heading':s['heading']} for s in sources],
             'reference_answer':case['reference_answer']}
        if generate:
            answer=assistant.answer(case['question'],provider=provider,model=model,api_key=api_key,top_k=top_k)
            row.update(answer=answer.text,answer_mode=answer.mode,notice=answer.notice,
                       abstained=answer.abstained,
                       reference_term_coverage=reference_coverage(answer.text,case.get('reference_groups',[])))
        row['latency_seconds']=round(time.perf_counter()-start,3)
        rows.append(row)
        if progress:
            progress((i+1)/len(cases))
    answerable=[r for r in rows if not r['expected_abstain']]
    coverages=[r['reference_term_coverage'] for r in rows if r.get('reference_term_coverage') is not None]
    return {'run_at':datetime.now(timezone.utc).isoformat(),'corpus_fetched_at':retriever.corpus['fetched_at'],
            'corpus_fingerprint':retriever.fingerprint,'retrieval_mode':retriever.mode,'top_k':top_k,
            'generation_requested':generate,'provider_requested':provider if generate else None,
            'metrics':{'answerable_cases':len(answerable),
                       'hit_at_k':sum(r['hit_at_k'] for r in answerable)/max(len(answerable),1),
                       'mrr_at_k':sum(r['reciprocal_rank'] for r in answerable)/max(len(answerable),1),
                       'mean_latency_seconds':sum(r['latency_seconds'] for r in rows)/max(len(rows),1),
                       'mean_reference_term_coverage':sum(coverages)/len(coverages) if coverages else None},
            'limitations':'Hit@k and MRR use curated URL AND heading labels. Reference-term coverage is a lexical proxy, not factual accuracy or semantic relevance. Citation ID checks do not establish entailment. No sample answer is indexed. Provided examples are development questions, not a held-out test.',
            'rows':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keyword-only', action='store_true')
    parser.add_argument('--split',choices=['provided','validation','all'],default='all')
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--provider',choices=['Source excerpts','Ollama','OpenAI'],default='Source excerpts')
    parser.add_argument('--model',default='')
    parser.add_argument('--output',type=Path,default=ROOT/'data'/'evaluation_report.json')
    args=parser.parse_args()
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    cases=[c for c in load_cases() if args.split=='all' or c['split']==args.split]
    retriever=Retriever(load_corpus(),semantic=not args.keyword_only)
    report=evaluate(retriever,cases,generate=args.generate,provider=args.provider,model=args.model,api_key=os.getenv('OPENAI_API_KEY',''))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['metrics'],indent=2))
    print(f"Saved {args.output}")


if __name__=='__main__':
    main()
