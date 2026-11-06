"""Heading-aware chunks, BM25 and persistent transformer/Chroma hybrid search."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ingest import ROOT, CORPUS_PATH

CACHE = ROOT / '.cache'
MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
STOP = set('a an the is are was were be to of in on at for and or with as by from it its this that these those what which how can i my me you your do does will would should there have has program programs university chicago uchicago master masters applied data science ms ads please tell about'.split())
ALIASES = {
    'tuition': 'cost price fees', 'cost': 'tuition fees', 'scholarship': 'funding financial aid',
    'scholarships': 'funding financial aid', 'deadline': 'application deadlines dates',
    'deadlines': 'application deadline dates', 'toefl': 'english language proficiency ielts',
    'ielts': 'english language proficiency toefl', 'visa': 'sponsorship international immigration',
    'stem': 'opt designated', 'courses': 'curriculum core elective capstone requirements',
    'advising': 'appointment counselor enrollment admissions', 'waiver': 'fee waivers',
    'transcripts': 'official transcript mail mailing address', 'decision': 'decisions released rolling',
    'mba': 'booth joint degree application',
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r'in[- ]person|on[- ]campus', 'inperson', text.lower())
    words = re.findall(r'[a-z0-9]+', text)
    return [w[:-1] if w.endswith('s') and len(w) > 4 else w for w in words if w not in STOP and len(w) > 1]


def expand_query(query: str) -> str:
    words = set(re.findall(r'[a-z]+', query.lower()))
    return query + ' ' + ' '.join(v for k, v in ALIASES.items() if k in words)


@dataclass(frozen=True)
class Chunk:
    id: str
    url: str
    title: str
    heading: str
    text: str
    fetched_at: str


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float = 0.0
    cosine: float | None = None

    def to_dict(self):
        return {**asdict(self.chunk), 'score': round(self.score, 5), 'bm25': round(self.bm25, 3),
                'cosine': round(self.cosine, 3) if self.cosine is not None else None}


def load_corpus(path: Path = CORPUS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError('Knowledge base not found. Run python -m rag.ingest from the app directory.')
    corpus = json.loads(path.read_text(encoding='utf-8'))
    if corpus.get('schema_version') != 1 or not corpus.get('pages'):
        raise ValueError('The knowledge-base file has an unsupported format or no pages.')
    return corpus


def make_chunks(corpus: dict) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=120,
                                             separators=['\n\n', '\n', '. ', ' ', ''])
    chunks, seen = [], set()
    for page in corpus['pages']:
        for section in page['sections']:
            for text in splitter.split_text(section['text']):
                digest = hashlib.sha256((page['url'] + '\n' + section['heading'] + '\n' + text).encode()).hexdigest()[:24]
                if digest in seen:
                    continue
                seen.add(digest)
                chunks.append(Chunk(digest, page['url'], page['title'], section['heading'], text, corpus['fetched_at']))
    return chunks


class BM25:
    def __init__(self, texts: list[str]):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.average = float(self.lengths.mean()) if len(self.lengths) else 1.0
        self.df = Counter(word for doc in self.docs for word in doc)

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(len(self.docs))
        for word in set(tokenize(query)):
            frequency = np.array([doc.get(word, 0) for doc in self.docs], dtype=float)
            count = self.df.get(word, 0)
            idf = math.log(1 + (len(self.docs) - count + .5) / (count + .5))
            norm = frequency + 1.5 * (1 - .75 + .75 * self.lengths / max(self.average, 1))
            result += idf * frequency * 2.5 / np.maximum(norm, 1e-9)
        return result


class Retriever:
    def __init__(self, corpus: dict, semantic: bool = True, allow_download: bool = False):
        self.corpus = corpus
        self.chunks = make_chunks(corpus)
        if not self.chunks:
            raise ValueError('No searchable chunks found in the corpus.')
        self.bm25 = BM25([c.heading + '\n' + c.heading + '\n' + c.title + '\n' + c.text for c in self.chunks])
        self.by_id = {c.id: c for c in self.chunks}
        self.model = self.collection = None
        self.semantic_error = None
        self.lock = threading.RLock()
        self.fingerprint = hashlib.sha256(('v1:' + MODEL_NAME + ':' + ':'.join(c.id for c in self.chunks)).encode()).hexdigest()[:20]
        if semantic:
            try:
                self._prepare_semantic(allow_download)
            except Exception as exc:
                # A visible keyword-only mode remains useful without downloads or a vector runtime.
                self.semantic_error = f'{type(exc).__name__}: {str(exc)[:250]}'
                self.model = self.collection = None

    @property
    def mode(self):
        return 'Hybrid · BM25 + MiniLM' if self.collection is not None else 'Keyword · BM25'

    def _prepare_semantic(self, allow_download: bool):
        os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
        os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        from sentence_transformers import SentenceTransformer
        import torch
        from chromadb import PersistentClient
        from chromadb.config import Settings
        CACHE.mkdir(exist_ok=True)
        torch.set_num_threads(max(1, min(int(os.getenv('RAG_CPU_THREADS', '2')), 8)))
        self.model = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE / 'models'),
                                         local_files_only=not allow_download, device='cpu')
        self.model.max_seq_length = 256
        self.client = PersistentClient(path=str(CACHE / 'chroma'), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection('msads_' + self.fingerprint,
                                                               metadata={'hnsw:space': 'cosine'}, embedding_function=None)
        indexed = set(self.collection.get(include=[])['ids'])
        missing = [c for c in self.chunks if c.id not in indexed]
        for start in range(0, len(missing), 48):
            batch = missing[start:start + 48]
            texts = [c.heading + '\n' + c.text for c in batch]
            vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.collection.upsert(ids=[c.id for c in batch], embeddings=vectors.tolist(),
                                   documents=[c.text for c in batch],
                                   metadatas=[{'url': c.url, 'heading': c.heading} for c in batch])

    def search(self, query: str, k: int = 6) -> list[Hit]:
        query = query.strip()[:3000]
        if not tokenize(query):
            return []
        expanded = expand_query(query)
        lexical = self.bm25.scores(expanded)
        lex_order = [int(i) for i in np.argsort(-lexical)[:50] if lexical[i] > 0]
        ranks, cosines = {}, {}
        for rank, index in enumerate(lex_order, 1):
            ranks[index] = 1 / (60 + rank)
        if self.collection is not None:
            with self.lock:
                vector = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
                matches = self.collection.query(query_embeddings=[vector.tolist()], n_results=min(50, len(self.chunks)), include=['distances'])
            indices = {c.id: i for i, c in enumerate(self.chunks)}
            for rank, (key, distance) in enumerate(zip(matches['ids'][0], matches['distances'][0]), 1):
                index = indices[key]
                cosine = 1 - float(distance)
                cosines[index] = cosine
                if cosine >= .20:
                    ranks[index] = ranks.get(index, 0) + 1 / (60 + rank)
        # Skip weak semantic-only matches; lexical scores are not calibrated probabilities.
        for index in ranks:
            scope = self.chunks[index].heading + ' ' + self.chunks[index].url
            if re.search(r'MBA|Booth|joint.degree', scope, re.I) and not re.search(r'\bmba\b|\bbooth\b|\bjoint\b', query, re.I):
                ranks[index] *= .35
            if re.search(r'\bPhD\b|doctoral', scope, re.I) and not re.search(r'\bphd\b|doctoral', query, re.I):
                ranks[index] *= .35
        order = sorted(ranks, key=ranks.get, reverse=True)
        result = []
        for index in order:
            if lexical[index] <= 0 and cosines.get(index, 0) < .35:
                continue
            result.append(Hit(self.chunks[index], ranks[index], float(lexical[index]), cosines.get(index)))
            if len(result) >= k:
                break
        return result
