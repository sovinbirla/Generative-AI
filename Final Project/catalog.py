"""Normalize the assigned catalog and optionally cache its listing photos."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import random
import re
from urllib.parse import urlparse
from urllib.request import urlopen

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
IMAGES = DATA / 'images'
FIELDS = {
    'brand': 'Brand Name', 'price': 'Selling Price', 'category': 'Category',
    'features': 'About Product', 'description': 'Product Description',
    'model': 'Model Number', 'color': 'Color', 'dimensions': 'Product Dimensions',
}


def clean(value):
    value = ' '.join((value or '').replace('\\xa0', ' ').split())
    return '' if value.casefold() in {'none', 'nan', 'n/a', 'null', '-'} else value


def prepare(limit, seed):
    with (DATA / 'amazon.csv').open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    random.Random(seed).shuffle(rows)
    products = []
    seen_ids, seen_titles = set(), set()
    for row in rows:
        title = clean(row['Product Name'])
        match = re.search(r'/dp/([A-Z0-9]{10})(?:[/?]|$)', row['Product Url'])
        pid = clean(row['Asin']) or (match[1] if match else clean(row['Uniq Id']))
        title_key = title.casefold()
        if not title or not re.fullmatch(r'[A-Za-z0-9]{1,64}', pid):
            continue
        if pid in seen_ids or title_key in seen_titles:
            continue
        seen_ids.add(pid)
        seen_titles.add(title_key)
        urls = list(dict.fromkeys(
            url for url in row['Image'].split('|')
            if urlparse(url).hostname in {'images-na.ssl-images-amazon.com', 'm.media-amazon.com'}
            and url.startswith('https://') and '/images/I/' in url
        ))
        product = {key: clean(row[column]) for key, column in FIELDS.items()}
        product['features'] = re.sub(
            r'^Make sure this fits by entering your model number\.\s*\|?\s*',
            '', product['features'], flags=re.I,
        )
        product.update(id=pid, title=title, url=clean(row['Product Url']), images=urls[:2])
        products.append(product)
    selected = products[:limit] if limit else products
    (DATA / 'catalog.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    (DATA / 'profile.json').write_text(json.dumps({
        'raw_rows': len(rows), 'unique_valid': len(products), 'selected': len(selected),
        'seed': seed, 'field_present': {
            key: sum(bool(item[key]) for item in selected) for key in FIELDS
        }, 'listed_image_urls': sum(bool(item['images']) for item in selected),
    }, indent=2))
    return selected


def image_path(product, view=0):
    return IMAGES / f"{product['id']}-{view}.jpg"


def download(task):
    product, view, url = task
    path = image_path(product, view)
    if path.exists():
        return True
    try:
        with urlopen(url, timeout=12) as response:
            with Image.open(BytesIO(response.read(8_000_000))) as source:
                if min(source.size) < 32:
                    return False
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((768, 768))
                image.save(path, quality=92)
                return True
    except (OSError, ValueError):
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=240, help='0 selects all unique records')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--download-images', action='store_true')
    args = parser.parse_args()
    products = prepare(args.limit, args.seed)
    if args.download_images:
        IMAGES.mkdir(exist_ok=True)
        tasks = [(p, i, url) for p in products for i, url in enumerate(p['images'])]
        with ThreadPoolExecutor(max_workers=16) as pool:
            print(f'Cached {sum(pool.map(download, tasks))}/{len(tasks)} listing images')
