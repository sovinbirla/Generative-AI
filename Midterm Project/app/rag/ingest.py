"""Bounded, robots-aware ingestion of the official MS-ADS website."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit, unquote
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

BASE = 'https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/'
ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / 'data' / 'corpus.json'
USER_AGENT = 'UChicago-MSADS-Course-RAG/1.0 (educational, bounded public-page retrieval)'


def canonical_url(url: str, parent: str = BASE) -> str | None:
    """Only permit public program pages and the two relevant official partners."""
    parts = urlsplit(urljoin(parent, url))
    if parts.scheme not in {'https', 'http'} or parts.username or parts.password:
        return None
    host = (parts.hostname or '').lower()
    path = re.sub('/+', '/', unquote(parts.path).strip())
    allowed = (
        host == 'datascience.uchicago.edu' and (
            path.startswith(urlsplit(BASE).path.rstrip('/'))
            or path.rstrip('/') in {
                '/education/masters-programs/in-person-program',
                '/education/masters-programs/online-program',
                '/education/tuition-fees-aid',
            })
        or host == 'www.chicagobooth.edu' and path.startswith('/mba/joint-degree/mba-ms-applied-data-science')
        or host == 'physicalsciences.uchicago.edu' and path.rstrip('/') == '/academics/application-requirements'
        or host == 'grad.uchicago.edu' and path.rstrip('/') == '/admissions/apply/english-language-requirements'
    )
    if not allowed or re.search(r'\.(pdf|jpg|png|zip|mp4|svg)$', path, re.I):
        return None
    return urlunsplit(('https', host, path.rstrip('/') + '/', '', ''))


def parse_page(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, 'html.parser')
    main = soup.find('main') or soup.find('article') or soup.body or soup
    links = sorted({u for a in main.find_all('a', href=True)
                    if (u := canonical_url(a['href'], url))})
    title_node = soup.find('h1') or soup.title
    title = title_node.get_text(' ', strip=True) if title_node else urlsplit(url).path
    for node in main.select('script, style, nav, footer, header, aside, form, svg, noscript, [class*="sidebar"]'):
        node.decompose()
    for question in main.select('a.accordion-title'):
        groups = []
        for parent in question.find_parents('li'):
            label = parent.find(['a', 'h3'], recursive=False)
            if label is not None and label is not question:
                groups.append(label.get_text(' ', strip=True))
        if groups:
            question.string = ' › '.join(list(reversed(groups)) + [question.get_text(' ', strip=True)])
        question.name = 'h3'
        question.attrs = {}
    # Preserve meaningful links (advising portals, applications and policies) as evidence.
    for a in main.find_all('a', href=True):
        href = urljoin(url, a['href'])
        label = a.get_text(' ', strip=True)
        if label and href.startswith(('https://', 'http://')):
            a.replace_with(f'{label} ({href})')
    sections, lines, heading, seen = [], [], title, set()

    def flush():
        if lines:
            sections.append({'heading': heading, 'text': '\n'.join(lines)})
            lines.clear()

    for tag in main.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'table', 'summary', 'dt', 'dd', 'div']):
        if tag.find_parent('table') is not None:
            continue
        if tag.name == 'li' and tag.find(['p', 'li', 'table', 'h3', 'h4', 'div']) is not None:
            continue
        if tag.name == 'div' and (not any(c in ['elementToProof', 'textblock'] for c in tag.get('class', []))
                                  or tag.find(['div', 'p', 'li', 'table', 'h1', 'h2', 'h3']) is not None):
            continue
        text = re.sub(r'[ \t]+', ' ', tag.get_text(' ', strip=True)).strip()
        text = re.sub(r'(?<=\d),{2,}(?=\d)', ',', text)
        # The website encodes pathway tuition as slash-separated price,label pairs.
        # Reorder those existing pairs into explicit labels without changing values.
        text = re.sub(r'(\$[\d,]+(?:\.\d+)?)([+*^]*),\s*(\d+)\s+courses', r'\3 courses: \1\2', text)
        text = re.sub(r'/\s*(?=\d+\s+courses:|\$\d)', '\n', text)
        if not text:
            continue
        if tag.name.startswith('h') and tag.name[1:].isdigit():
            if re.search(r'related (news|events)|upcoming events|stay connected', text, re.I):
                break
            flush()
            heading = text
            continue
        if tag.name == 'table':
            text = '\n'.join(' | '.join(cell.get_text(' ', strip=True) for cell in row.find_all(['th', 'td'])) for row in tag.find_all('tr'))
            text = re.sub(r'(?<=\d),{2,}(?=\d)', ',', text)
        normalized = re.sub(r'\s+', ' ', text).lower()
        if normalized not in seen and len(text) > 12:
            seen.add(normalized)
            lines.append(text)
    flush()
    return {'url': url, 'title': title, 'sections': sections, 'links': links,
            'content_hash': hashlib.sha256(json.dumps(sections, sort_keys=True).encode()).hexdigest()}


def crawl(max_pages: int = 28, delay: float = .25) -> dict:
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT
    queue, visited, pages, errors, robots = deque([BASE]), set(), [], [], {}
    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        origin = 'https://' + urlsplit(url).netloc
        try:
            if origin not in robots:
                response = session.get(origin + '/robots.txt', timeout=20)
                if response.status_code in (404, 410):
                    parser = RobotFileParser()
                    parser.parse([])
                else:
                    response.raise_for_status()
                    parser = RobotFileParser()
                    parser.parse(response.text.splitlines())
                robots[origin] = parser
            if not robots[origin].can_fetch(USER_AGENT, url):
                errors.append({'url': url, 'error': 'Disallowed by robots.txt'})
                continue
            response = session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            resolved = canonical_url(response.url)
            if resolved is None:
                errors.append({'url': url, 'error': 'Redirect outside the allowed corpus'})
                continue
            if 'text/html' not in response.headers.get('Content-Type', ''):
                continue
            raw_dir = ROOT / '.cache' / 'raw'
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / (hashlib.sha256(resolved.encode()).hexdigest()[:20] + '.html')).write_text(response.text, encoding='utf-8')
            page = parse_page(response.text, resolved)
            if not page['sections']:
                errors.append({'url': url, 'error': 'No main-page text extracted'})
                continue
            if not any(p['url'] == resolved for p in pages):
                pages.append(page)
                print(f"Collected {len(pages):02d}: {page['title']} ({sum(len(s['text']) for s in page['sections'])} chars)", flush=True)
            for link in page['links']:
                if link not in visited and link not in queue:
                    queue.append(link)
        except (requests.RequestException, ValueError) as exc:
            errors.append({'url': url, 'error': f'{type(exc).__name__}: {str(exc)[:180]}'})
        time.sleep(delay)
    return {'schema_version': 1, 'source': BASE, 'fetched_at': datetime.now(timezone.utc).isoformat(),
            'pages': pages, 'errors': errors, 'unvisited_links': list(queue),
            'note': 'Only official website text is answer evidence. Assignment sample answers are evaluation references, not knowledge-base content.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-pages', type=int, default=28)
    args = parser.parse_args()
    if not 1 <= args.max_pages <= 60:
        parser.error('--max-pages must be between 1 and 60')
    corpus = crawl(args.max_pages)
    if len(corpus['pages']) < 4:
        raise SystemExit('Fewer than four official pages collected; existing corpus was not replaced. Check connectivity and crawl errors: ' + json.dumps(corpus['errors']))
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CORPUS_PATH.with_suffix('.tmp')
    temporary.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(CORPUS_PATH)
    print(f"Saved {len(corpus['pages'])} pages; {len(corpus['errors'])} crawl notices to {CORPUS_PATH}")


if __name__ == '__main__':
    main()
