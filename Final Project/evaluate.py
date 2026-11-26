"""Run deterministic retrieval ablations and QA on the prepared catalog."""
import argparse
from collections import defaultdict
import csv
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import random
import re
from statistics import mean
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer

from catalog import DATA, ROOT, image_path
from rag import DEVICE, MODES, Store, answer, description

OUT = ROOT / 'results'


def photo(product, view=0):
    with Image.open(image_path(product, view)) as source:
        return source.convert('RGB')


def transformed(image):
    inset = max(1, min(image.size) // 20)
    cropped = image.crop((inset, inset, image.width - inset, image.height - inset))
    return ImageOps.expand(cropped, border=inset, fill=(235, 235, 235))


def different_view(first, alternate):
    thumb = lambda x: np.asarray(ImageOps.pad(x, (64, 64), color='white'), dtype=np.float32)
    return np.abs(thumb(first) - thumb(alternate)).mean() > 3


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def version(store):
    model_sources = {}
    for key, env, directory in [('clip', 'CLIP_MODEL', 'clip'), ('qwen', 'QA_MODEL', 'qwen')]:
        source = os.getenv(env, str(ROOT / 'models' / directory))
        path = Path(source)
        files = [p for p in path.iterdir() if p.suffix in {'.bin', '.safetensors'}] if path.is_dir() else []
        model_sources[key] = {
            'source': str(path.resolve().relative_to(ROOT)) if path.is_dir()
                      and path.resolve().is_relative_to(ROOT) else source,
            'weight_sha256': {file.name: digest(file) for file in sorted(files)},
        }
    return {'device': DEVICE, 'architecture': platform.machine(),
            'python': platform.python_version(), 'packages': {
                name: package_version(name) for name in ('torch', 'transformers', 'numpy', 'scikit-learn', 'Pillow')
            }, 'models': model_sources, 'catalog_sha256': digest(DATA / 'catalog.json'),
            'raw_catalog_sha256': digest(DATA / 'amazon.csv'),
            'index_sha256': digest(DATA / 'catalog.sqlite'),
            'indexed_products': len(store.products), 'indexed_images': int(store.has_image.sum())}


def retrieval(store):
    generator = random.Random(9)
    sample = store.products.copy()
    generator.shuffle(sample)
    sparse = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    sparse_matrix = sparse.fit_transform(description(p) for p in store.products)
    rankings = []

    def log(task, method, item, ids, milliseconds, query=''):
        rank = ids.index(item['id']) + 1 if item['id'] in ids else None
        rankings.append(dict(task=task, method=method, id=item['id'],
                             query=query, rank=rank, top_ids=' '.join(ids),
                             ms=round(milliseconds, 3)))

    def run(task, method, item, text='', image=None, **options):
        start = perf_counter()
        hits = store.search(text, image, k=10, **options)
        log(task, method, item, [p['id'] for p, _ in hits], (perf_counter() - start) * 1000, text)

    # One weakly labeled title-derived query per listing, with deterministic token deletion.
    for item in sample[:80]:
        words = item['title'].split()
        query = ' '.join(w for i, w in enumerate(words[:14]) if i % 3 != 2)
        start = perf_counter()
        scores = (sparse_matrix @ sparse.transform([query]).T).toarray().ravel()
        indices = np.argsort(-scores, kind='stable')[:10]
        log('text', 'TF-IDF rich', item, [store.products[i]['id'] for i in indices],
            (perf_counter() - start) * 1000, query)
        run('text', 'CLIP title', item, query, representation='title')
        run('text', 'CLIP rich', item, query)

    for item in [p for p in sample if image_path(p).exists()][:60]:
        primary = photo(item)
        image = transformed(primary)
        run('transformed', 'CLIP cross-modal', item, image=image, use_image_index=False)
        run('transformed', 'CLIP visual', item, image=image)
        if image_path(item, 1).exists():
            alternate = photo(item, 1)
            if different_view(primary, alternate):
                query = 'Tell me about this ' + (item['category'].split('|')[-1].strip() or 'product')
                run('alternate', 'CLIP cross-modal', item, image=alternate, use_image_index=False)
                run('alternate', 'CLIP visual', item, image=alternate)
                run('alternate', 'CLIP category text', item, query)
                run('alternate', 'CLIP fused', item, query, alternate)

    groups = defaultdict(list)
    for row in rankings:
        groups[row['task'], row['method']].append(row)
    summary = []
    for (task, method), rows in groups.items():
        ranks = [r['rank'] or float('inf') for r in rows]
        summary.append(dict(task=task, method=method, n=len(rows), **{
            f'recall_at_{k}': mean(r <= k for r in ranks) for k in [1, 5, 10]
        }, mrr_at_10=mean(1 / r for r in ranks), median_ms=float(np.median([r['ms'] for r in rows]))))
    write_csv('retrieval_cases.csv', rankings)
    write_csv('retrieval_metrics.csv', summary)
    print(json.dumps(summary, indent=2), flush=True)


def qa(store, per_slice):
    rng = random.Random(17)
    priced = [p for p in store.products if re.fullmatch(r'\$[\d,]+\.\d{2}', p['price'])]
    rng.shuffle(priced)
    alternate = []
    for p in priced:
        if image_path(p).exists() and image_path(p, 1).exists():
            image = photo(p, 1)
            if different_view(photo(p), image):
                alternate.append(p)
    text_items = [p for p in priced if p['id'] not in {a['id'] for a in alternate[:per_slice]}]
    warranty_items = [p for p in text_items[per_slice:]
                      if 'warranty' not in description(p).lower()]
    cases = []
    for task, items in [('text_price', text_items), ('image_price', alternate),
                        ('missing_warranty', warranty_items)]:
        for p in items[:per_slice]:
            if task == 'image_price':
                query = 'What is the listed price of this product?'
                hits = store.search(query, photo(p, 1), k=3)
            elif task == 'text_price':
                query = f"What is the listed price of {p['title']}?"
                hits = store.search(p['title'], k=3)
            else:
                query = f"How many years of warranty does {p['title']} include?"
                hits = store.search(p['title'], k=3)
            products = [x for x, _ in hits]
            if task == 'missing_warranty' and any('warranty' in description(x).lower() for x in products):
                continue
            cases.append((task, p, query, products))
    rows = []
    for task, p, query, products in cases:
        ids = [x['id'] for x in products]
        expected = p['price'] if task != 'missing_warranty' else 'catalog omission'
        for mode in MODES:
            start = perf_counter()
            output = answer(query, products, mode)
            seconds = perf_counter() - start
            refs = [int(n) for n in re.findall(r'\[P(\d+)\]', output)]
            amounts = re.findall(r'\$\s*[\d,.]+', output)
            amounts = {re.sub(r'[,\s]', '', x.rstrip('.,')) for x in amounts}
            abstained = bool(re.search(
                r"not (?:say|mention|provide|specify|include|state|list|available|given|specified|mentioned|provided|stated|listed|included|found)|no .*warranty information|doesn't (?:say|mention|provide|specify|state|list|include)|(?:cannot|could not) (?:determine|find|confirm)|no information",
                output, flags=re.I,
            ))
            correct = ((abstained and not re.search(r'\b\d+\s*[- ]?years?\b', output))
                       if task == 'missing_warranty' else amounts == {p['price'].replace(',', '')})
            cites_target = p['id'] in ids and ids.index(p['id']) + 1 in refs
            valid_refs = all(1 <= ref <= len(ids) for ref in refs)
            relevant = bool(re.search(r'warranty', output, re.I)) if task == 'missing_warranty' else bool(amounts)
            row = dict(task=task, mode=mode, id=p['id'], query=query, expected=expected,
                       retrieved=' '.join(ids), gold_in_context=int(p['id'] in ids), response=output,
                       relevant=int(relevant), correct=int(correct), cited_target=int(cites_target),
                       grounded_price=int(task != 'missing_warranty' and correct and cites_target and valid_refs),
                       seconds=round(seconds, 3))
            rows.append(row)
            print(task, p['id'], mode, row['correct'], round(seconds, 2), flush=True)
    write_csv('qa_cases.csv', rows)
    summary = []
    for task in ('all', 'text_price', 'image_price', 'missing_warranty'):
        for mode in MODES:
            subset = [row for row in rows if row['mode'] == mode and task in {'all', row['task']}]
            if not subset:
                continue
            prices = [row for row in subset if row['task'] != 'missing_warranty']
            summary.append(dict(task=task, mode=mode, n=len(subset),
                relevance_proxy=mean(row['relevant'] for row in subset),
                correctness_proxy=mean(row['correct'] for row in subset),
                gold_in_context=mean(row['gold_in_context'] for row in subset),
                grounded_price=mean(row['grounded_price'] for row in prices) if prices else '',
                median_seconds=float(np.median([row['seconds'] for row in subset]))))
    write_csv('qa_metrics.csv', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['retrieval', 'qa', 'all'], default='all')
    parser.add_argument('--qa-per-slice', type=int, default=6)
    args = parser.parse_args()
    store = Store()
    (OUT / 'run.json').write_text(json.dumps(version(store), indent=2))
    if args.stage in ('all', 'retrieval'):
        retrieval(store)
    if args.stage in ('all', 'qa'):
        qa(store, args.qa_per_slice)
