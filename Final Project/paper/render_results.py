"""Generate LaTeX tables and result statements from the recorded run."""
import csv
import json
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
SLASH = chr(92)
ROW_END = ' ' + SLASH * 2
TASKS = {
    'text': 'Title-derived', 'transformed': 'Transformed photo',
    'alternate': 'Alternate photo', 'all': 'All',
    'text_price': 'Title price', 'image_price': 'Photo price',
    'missing_warranty': 'Missing warranty',
}
MODES = {'zero': 'Zero-shot', 'few': 'Two-shot', 'multi': 'Four-shot'}


def csv_rows(name):
    with (RESULTS / name).open(newline='') as file:
        return list(csv.DictReader(file))


def percent(value):
    return f'{100 * float(value):.1f}'


def pct(value):
    return percent(value) + SLASH + '%'


def wilson(value, n):
    rate, n, z = float(value), int(n), 1.96
    denominator = 1 + z ** 2 / n
    center = (rate + z ** 2 / (2 * n)) / denominator
    radius = z * sqrt(rate * (1 - rate) / n + z ** 2 / (4 * n ** 2)) / denominator
    return f'{percent(center - radius)}--{pct(center + radius)}'


def table(rows):
    return '\n'.join(' & '.join(str(field) for field in row) + ROW_END for row in rows)


def escape(value):
    replacements = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
        '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
    }
    return ''.join(replacements.get(char, char) for char in value)


def main():
    profile = json.loads((ROOT / 'data/profile.json').read_text())
    run = json.loads((RESULTS / 'run.json').read_text())
    retrieval = csv_rows('retrieval_metrics.csv')
    qa = csv_rows('qa_metrics.csv')
    cases = csv_rows('qa_cases.csv')
    rr = {(r['task'], r['method']): r for r in retrieval}
    qr = {(r['task'], r['mode']): r for r in qa}
    text_sparse = rr['text', 'TF-IDF rich']
    text_title = rr['text', 'CLIP title']
    text_rich = rr['text', 'CLIP rich']
    crop_direct = rr['transformed', 'CLIP visual']
    crop_cross = rr['transformed', 'CLIP cross-modal']
    alt_direct = rr['alternate', 'CLIP visual']
    alt_cross = rr['alternate', 'CLIP cross-modal']
    alt_fused = rr['alternate', 'CLIP fused']
    accuracy = lambda r: pct(r['recall_at_1'])
    all_rates = [pct(qr['all', mode]['correctness_proxy']) for mode in MODES]
    n = qr['all', 'zero']['n']
    image_rates = [pct(qr['image_price', mode]['correctness_proxy']) for mode in MODES]
    evidence_rate = pct(qr['image_price', 'zero']['gold_in_context'])

    failure = next((c for c in cases if c['task'] == 'image_price'
                    and c['correct'] == '0' and c['gold_in_context'] == '0'), None)
    if failure:
        quoted = escape(' '.join(failure['response'].split())[:110])
        missing_case = (
            f"On an alternate-photo price query for {failure['id']}, the reference price "
            f"was {escape(failure['expected'])}. The {MODES[failure['mode']].lower()} model "
            f"answered with the excerpt \\emph{{{quoted}}}. The target listing was absent "
            "from the supplied evidence."
        )
    else:
        missing_case = ''
    available_failure = next((c for c in cases if c['task'] == 'image_price'
                              and c['correct'] == '0' and c['gold_in_context'] == '1'), None)
    if available_failure:
        quoted = escape(' '.join(available_failure['response'].split())[:105])
        rating = MODES[available_failure['mode']].lower()
        missing_case += (
            f" For {available_failure['id']}, the target was among the supplied records and "
            f"listed at {escape(available_failure['expected'])}. The {rating} response "
            f"included \\emph{{{quoted}}} and failed the exact-price check."
        )
    coverage = {
        'Selling price': 'price', 'Category': 'category', 'Features': 'features',
        'Model number': 'model', 'Dimensions': 'dimensions', 'Brand': 'brand',
        'Color': 'color', 'Stand-alone description': 'description',
    }
    specs = {
        'RawRows': f"{profile['raw_rows']:,}",
        'UniqueRows': f"{profile['unique_valid']:,}",
        'CoverageRows': table([(label, profile['field_present'][key]) for label, key in coverage.items()]
                              + [('Indexed primary photo', run['indexed_images'])]),
        'AbstractResults': (
            f"Visual search reached {accuracy(crop_direct)} Recall@1 on 60 transformed primary "
            f"photos and {accuracy(alt_direct)} on {alt_direct['n']} alternate listing photos. "
            f"Category-text fusion reached {accuracy(alt_fused)} on the alternate slice."
        ),
        'AbstractQA': (
            f"Across {n} answer cases per prompt, exact-price and abstention checks passed on "
            f"{all_rates[0]}, {all_rates[1]}, and {all_rates[2]} of cases, respectively."
        ),
        'RetrievalDiscussion': (
            f"TF-IDF attained {accuracy(text_sparse)} Recall@1 on the title-derived queries. "
            f"Enriching titles moved CLIP Recall@1 from {accuracy(text_title)} to "
            f"{accuracy(text_rich)}, while improving Recall@10 from {pct(text_title['recall_at_10'])} "
            f"to {pct(text_rich['recall_at_10'])}. Direct visual indexing improved transformed-photo "
            f"Recall@1 from {accuracy(crop_cross)} to {accuracy(crop_direct)}. With alternate photos, "
            f"direct visual and cross-modal retrieval attained {accuracy(alt_direct)} and "
            f"{accuracy(alt_cross)} Recall@1, respectively. Fusion with a category cue attained "
            f"{accuracy(alt_fused)}. The descriptive 95\\% Wilson interval for alternate-photo "
            f"visual Recall@1 was {wilson(alt_direct['recall_at_1'], alt_direct['n'])}. "
            "The interval describes binomial uncertainty on this slice and does not address "
            "dataset shift."
        ),
        'QADiscussion': (
            f"The cumulative correctness proxy was {all_rates[0]} for zero shots, "
            f"{all_rates[1]} for two shots, and {all_rates[2]} for four shots. "
            f"For photo price questions the corresponding rates were {image_rates[0]}, "
            f"{image_rates[1]}, and {image_rates[2]}. The reference listing reached the prompt "
            f"in {evidence_rate} of photo price cases, limiting what prompt examples could recover."
        ),
        'CaseDiscussion': missing_case,
        'ConclusionResults': (
            "Indexed photos worked well on modified copies but showed a substantial drop on "
            "alternate views. On the benchmark, broad category-text fusion further reduced "
            "top-1 alternate-view accuracy."
        ),
        'RetrievalRows': table([
            [TASKS[r['task']], r['method'], r['n'],
             *(percent(r[k]) for k in ('recall_at_1', 'recall_at_5', 'recall_at_10', 'mrr_at_10')),
             f"{float(r['median_ms']):.1f}"] for r in retrieval
        ]),
        'QARows': table([
            [TASKS[r['task']], MODES[r['mode']], r['n'],
             *(percent(r[k]) if r[k] else '--' for k in
               ('relevance_proxy', 'correctness_proxy', 'gold_in_context', 'grounded_price')),
             f"{float(r['median_seconds']):.1f}"] for r in qa
        ]),
    }
    (ROOT / 'paper/measured.tex').write_text('\n'.join(
        f'\\newcommand{{\\{key}}}{{{value}}}' for key, value in specs.items()
    ) + '\n')


if __name__ == '__main__':
    main()
