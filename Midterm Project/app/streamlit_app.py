"""Streamlit entry point: run with streamlit run streamlit_app.py."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from rag.engine import Assistant, list_ollama_models
from rag.evaluation import evaluate, load_cases
from rag.ingest import ROOT, CORPUS_PATH
from rag.retrieval import Retriever, load_corpus

load_dotenv(ROOT / '.env', override=False)
st.set_page_config(page_title='MS-ADS | Program Guide', page_icon='🎓', layout='wide', initial_sidebar_state='auto')
st.markdown('''<style>
+.stApp{background:#faf9f7} .block-container{max-width:1120px;padding-top:2.1rem;padding-bottom:4rem}
+[data-testid="stSidebar"]{background:#f0ede8;border-right:1px solid #e1dcd4}
+.brand{font-size:12px;letter-spacing:.15em;font-weight:750;color:#800000;margin-bottom:8px}
+.hero{background:#800000;color:#fff;padding:30px 34px;border-radius:16px;margin-bottom:20px}
+.hero .kicker{font-size:11px;letter-spacing:.17em;text-transform:uppercase;color:#e1c3c3}
+.hero h1{font-size:36px;line-height:1.15;letter-spacing:-.035em;margin:10px 0 12px;color:#fff}
+.hero p{font-size:15px;line-height:1.65;color:#f4e8e8;max-width:780px;margin:0}
+.source-meta{font-size:12px;color:#62717b;margin-bottom:20px}
+[data-testid="stChatMessage"]{border:1px solid #e5e0d9;border-radius:12px;padding:20px;background:#fff;margin-bottom:14px}
+div[data-testid="stMetric"]{background:#fff;border:1px solid #e5e0d9;border-radius:12px;padding:12px 16px}
+div[data-testid="stMetricValue"]{font-size:25px}
+button[kind="secondary"]{border-color:#ded7cd}
+@media(max-width:700px){.hero{padding:24px}.hero h1{font-size:29px}.block-container{padding-top:1rem}}
+</style>'''.replace('\n+', '\n'), unsafe_allow_html=True)


def secret(name: str, default=''):
    try:
        return os.getenv(name) or st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        return os.getenv(name, default)


@st.cache_resource(show_spinner='Preparing the program knowledge base…')
def search_resource(fingerprint: str, semantic: bool, allow_download: bool):
    return Retriever(load_corpus(), semantic=semantic,
                     allow_download=allow_download)


@st.cache_data(ttl=15, show_spinner=False)
def local_models():
    return list_ollama_models()


def render_answer(message):
    st.markdown(message['content'])
    data = message.get('details', {})
    if data.get('notice'):
        st.info(data['notice'])
    if data.get('pii_redacted'):
        st.caption('Email addresses, phone numbers or SSN patterns were removed from your question before processing.')
    if data:
        st.caption(f"{data['mode']} · {data['retrieval_mode']} · {data['latency_seconds']:.1f}s")
    sources=data.get('sources', [])
    if sources:
        with st.expander(f'Inspect {len(sources)} retrieved passages'):
            st.caption('These are the passages supplied to the answer step. Retrieval scores are rankings, not confidence probabilities.')
            for n,source in enumerate(sources,1):
                st.markdown(f"**[{n}] {source['title']} — {source['heading']}**")
                st.link_button('Open official page', source['url'])
                st.text(source['text'])
                score=f"BM25 {source['bm25']:.2f}"
                if source['cosine'] is not None:
                    score+=f" · cosine similarity {source['cosine']:.3f}"
                st.caption(f"Saved {source['fetched_at'][:10]} · {score}")
                if n<len(sources): st.divider()


try:
    corpus=load_corpus()
except (OSError, ValueError) as exc:
    st.error(str(exc)); st.code('python -m rag.ingest', language='bash'); st.stop()

with st.sidebar:
    st.markdown('<div class="brand">UCHICAGO · DATA SCIENCE</div>', unsafe_allow_html=True)
    st.title('Program guide')
    st.caption('MS in Applied Data Science')
    page=st.radio('Explore', ['Ask the guide', 'Source library', 'Evaluation'], label_visibility='collapsed')
    st.divider()
    models=local_models()
    server_key=secret('OPENAI_API_KEY')
    available=['Source excerpts','OpenAI'] + (['Ollama'] if models else [])
    preferred='OpenAI' if server_key else ('Ollama' if models else 'Source excerpts')
    with st.expander('Answer settings', expanded=not bool(st.session_state.get('messages'))):
        provider=st.selectbox('Answer mode',available,index=available.index(preferred))
        api_key=server_key
        model=''
        if provider=='OpenAI':
            if not server_key:
                api_key=st.text_input('OpenAI API key',type='password',help='Used only in this browser session. It is not written to disk or included in chat downloads.')
            model=st.text_input('OpenAI model',value=secret('OPENAI_MODEL','gpt-4.1-mini'))
            st.caption('Your redacted question and retrieved public passages will be sent to OpenAI.')
        elif provider=='Ollama':
            preferred_model=os.getenv('OLLAMA_MODEL','')
            model=st.selectbox('Local model',models,index=models.index(preferred_model) if preferred_model in models else 0)
            st.caption('Generated answers use the Ollama server on this machine.')
        else:
            st.caption('Returns cited source excerpts. No language model is called.')
        semantic=st.toggle('Semantic + keyword retrieval',value=True)
        top_k=st.slider('Retrieved passages',min_value=3,max_value=10,value=6)
    if st.button('Start a new conversation', width='stretch'):
        st.session_state.messages=[]
        st.session_state.pop('pending_question', None)
        st.rerun()
    st.caption('Unofficial course project. Answers are based on a dated copy of official program pages. Confirm time-sensitive details with admissions.')

fingerprint=hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
retriever=search_resource(fingerprint, semantic, str(secret('ALLOW_MODEL_DOWNLOAD','0')).lower() in {'1','true'})
assistant=Assistant(retriever)
st.session_state.setdefault('messages', [])

st.markdown('''<header class="hero"><div class="kicker">The University of Chicago · MS in Applied Data Science</div><h1>Your next chapter starts with a question.</h1><p>Explore admissions, tuition, curriculum and student life—with answers you can trace back to official program sources.</p></header>''',unsafe_allow_html=True)
saved=datetime.fromisoformat(corpus['fetched_at']).strftime('%b %d, %Y')
st.caption(f"{len(corpus['pages'])} official pages · Snapshot: {saved} · {retriever.mode}")
if retriever.semantic_error:
    with st.expander('About keyword search'):
        st.write('Keyword search is available. The local semantic model or vector index could not be loaded. To prepare semantic search, run this command from the app folder:')
        st.code('python -m rag.prepare --download-model',language='bash')
        st.caption(retriever.semantic_error)

if page=='Ask the guide':
    if not st.session_state.messages:
        st.subheader('What would you like to know?')
        st.write('Ask in your own words, or start with one of these questions.')
        starters=[('Tuition & funding','What is tuition cost for the program?'),
                  ('Application deadlines','What are the deadlines for the in-person program?'),
                  ('English requirements','What are the minimum TOEFL and IELTS scores?'),
                  ('Curriculum & courses','How many courses are required, and is there a thesis track?'),
                  ('Online & visas','Does the online program provide visa sponsorship?'),
                  ('Meet an adviser','Can I schedule an appointment with an admissions counselor?')]
        columns=st.columns(2)
        for n,(label,question) in enumerate(starters):
            if columns[n%2].button(label,width='stretch',key=f'starter_{n}'):
                st.session_state.pending_question=question
                st.rerun()
        st.caption('Please leave out personal records, application IDs and contact details.')
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message['role']=='assistant': render_answer(message)
            else: st.markdown(message['content'])
    disabled=provider=='OpenAI' and not api_key
    if disabled:
        st.info('Add an API key in Answer settings, or choose Source excerpts to continue without a key.')
    typed=st.chat_input('Ask about the MS in Applied Data Science…',max_chars=2000,disabled=disabled)
    question=typed or st.session_state.pop('pending_question',None)
    if question and not disabled:
        with st.spinner('Finding the right program sources…'):
            answer=assistant.answer(question,st.session_state.messages,provider=provider,api_key=api_key,
                                    model=model,top_k=top_k)
        st.session_state.messages.append({'role':'user','content':answer.question})
        st.session_state.messages.append({'role':'assistant','content':answer.text,'details':asdict(answer)})
        st.rerun()
    if st.session_state.messages:
        st.download_button('Download conversation',json.dumps(st.session_state.messages,ensure_ascii=False,indent=2),
                           file_name='msads_conversation.json',mime='application/json')

elif page=='Source library':
    st.subheader('Every answer starts with evidence')
    a,b,c=st.columns(3)
    a.metric('Official pages',len(corpus['pages']))
    b.metric('Searchable passages',len(retriever.chunks))
    c.metric('Snapshot date',corpus['fetched_at'][:10])
    st.write('The assignment brief and sample answers are excluded from the search index. This library contains only saved official website text.')
    query=st.text_input('Filter sources by title or URL')
    pages=[p for p in corpus['pages'] if query.lower() in (p['title']+' '+p['url']).lower()]
    for p in pages:
        with st.expander(f"{p['title']} · {len(p['sections'])} sections"):
            st.link_button('View official page',p['url'])
            for section in p['sections']:
                st.markdown('**'+section['heading']+'**')
                st.text(section['text'])
    with st.expander('Collection details'):
        st.write('Public program pages and relevant official policy pages were collected with a bounded, robots-aware crawler. Expandable FAQ text, headings, tables and links are preserved. A maintainer refreshes the snapshot; questions do not trigger live browsing.')
        st.json({'fetched_at':corpus['fetched_at'],'crawl_notices':corpus['errors'],
                 'unvisited_links':corpus.get('unvisited_links',[]),'index_fingerprint':retriever.fingerprint})
        st.code('python -m rag.ingest\npython -m rag.prepare --download-model',language='bash')

else:
    st.subheader('Measure retrieval, inspect the answers')
    st.write('Run the twelve supplied development questions or the additional validation checks against this exact knowledge-base snapshot.')
    selected=st.selectbox('Question set',['Provided examples','Additional validation','All questions'])
    split={'Provided examples':'provided','Additional validation':'validation','All questions':'all'}[selected]
    cases=[case for case in load_cases() if split=='all' or case['split']==split]
    generate=st.checkbox('Include answers using the selected answer mode',value=False,
                         help='This makes a model call for each question when OpenAI or Ollama is selected.')
    st.caption('Hit@k: fraction of answerable questions with a relevant passage in the top k. MRR@k: rewards the first relevant passage appearing nearer the top. Relevance labels require both the correct source page and section heading.')
    if st.button('Run evaluation',type='primary',disabled=generate and provider=='OpenAI' and not api_key):
        progress=st.progress(0.0)
        with st.spinner('Evaluating retrieval and optional answers…'):
            report=evaluate(retriever,cases,top_k=top_k,generate=generate,provider=provider,model=model,
                            api_key=api_key,progress=progress.progress)
        st.session_state.evaluation=report
        progress.empty()
    if 'evaluation' in st.session_state:
        report=st.session_state.evaluation
        st.caption(f"Report from {report['run_at'][:19]} UTC · {report['retrieval_mode']} · top {report['top_k']} · {len(report['rows'])} questions")
        a,b,c=st.columns(3)
        a.metric(f"Hit@{report['top_k']}",f"{report['metrics']['hit_at_k']:.0%}")
        b.metric(f"MRR@{report['top_k']}",f"{report['metrics']['mrr_at_k']:.3f}")
        c.metric('Mean latency',f"{report['metrics']['mean_latency_seconds']:.2f}s")
        st.dataframe([{'Question':r['question'],'Relevant passage found':r['hit_at_k'],
                       'First relevant rank':r['first_relevant_rank'],'Seconds':r['latency_seconds']} for r in report['rows']],hide_index=True,width='stretch')
        for row in report['rows']:
            with st.expander(row['question']):
                st.markdown('**Snapshot-checked reference:** '+row['reference_answer'])
                if 'answer' in row:
                    st.markdown(row['answer'])
                    st.caption(f"Actual answer mode: {row['answer_mode']} · Reference-term coverage (lexical proxy): {row.get('reference_term_coverage')}")
                    if row['notice']: st.info(row['notice'])
                st.json(row['sources'])
        st.info(report['limitations'])
        st.download_button('Download evaluation report',json.dumps(report,ensure_ascii=False,indent=2),
                           file_name='msads_evaluation.json',mime='application/json')
    st.caption('For a response-relevance assessment, manually score correctness, completeness, citation support and format/year handling using the rubric in README.md. Automated keyword coverage does not establish factual accuracy.')
