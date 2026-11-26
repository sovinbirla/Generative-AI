"""Find supported catalog matches before offering any listing as evidence."""
import json
import re
from functools import cached_property

import numpy as np
from PIL import Image, ImageStat
from sklearn.feature_extraction.text import TfidfVectorizer

from catalog import DATA, image_path, load_products
from rag import encode

REFERENCES = DATA / 'product_references.json'
QUESTION_WORDS = frozenset('''
    a about an and are can compare cost could describe display do does explain feature features
    find for give has have how i identify image information is it its listed listing look
    in me my name of on photo picture please price product release resolution s see show spec specs
    specifications tell the this to use used usage warranty what when where which will with you
'''.split())
VARIANTS = {'fe', 'max', 'mini', 'plus', 'pro', 'ultra'}


def words(value):
    normalized = re.sub(r'\bfan edition\b', 'fe', value.casefold())
    return re.findall(r'\b[^\W_]+\b', re.sub(r'(?<=\w)\+', ' plus', normalized))


def query_terms(query):
    # Remove dollar amounts so a budget is not mistaken for a model identifier.
    return [w for w in words(re.sub(r'\$\s*[\d,.]+', '', query)) if w not in QUESTION_WORDS]


def model_terms(terms):
    return {w for w in terms if w in VARIANTS or any(char.isdigit() for char in w)}


class CatalogLookup:
    def __init__(self, image_store, source=DATA / 'amazon.csv'):
        self.image_store = image_store
        historical = load_products(path=source)[0] if source.is_file() else image_store.products
        self.references = json.loads(REFERENCES.read_text())
        self.historical_count = len(historical)
        self.products = self.references + historical
        names = [' '.join([p['title'], *p.get('aliases', [])]) for p in self.products]
        self.name_tokens = [set(words(name)) for name in names]
        self.text = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
        self.text_index = self.text.fit_transform(' '.join(words(name)) for name in names)

    def find_text(self, query, k=1):
        terms = query_terms(query)
        if not terms:
            return []
        needed_models = model_terms(terms)
        query_words = set(terms)
        encoded = self.text.transform([' '.join(terms)])
        scores = (self.text_index @ encoded.T).toarray().ravel()
        hits = []
        for index in np.argsort(-scores, kind='stable'):
            if scores[index] < 0.16:
                break
            name = self.name_tokens[index]
            if not needed_models <= name or len(query_words & name) < min(2, len(query_words)):
                continue
            product = self.products[index]
            if product.get('source_name'):
                aliases = [product['title'], *product.get('aliases', [])]
                if not any(set(words(alias)) <= query_words for alias in aliases):
                    continue
            hits.append(product)
            if len(hits) == k:
                break
        return hits

    @cached_property
    def visual_index(self):
        store = self.image_store
        products = [p for p, available in zip(store.products, store.has_image) if available]
        vectors = store.image[store.has_image]
        references, photos = [], []
        for product in self.references:
            path = image_path(product)
            if path.is_file():
                with Image.open(path) as source:
                    photos.append(source.convert('RGB'))
                references.append(product)
        if photos:
            vectors = np.concatenate((vectors, encode(photos, 'image')))
        return products + references, vectors

    def find_photo(self, image):
        image = image.convert('RGB')
        if ImageStat.Stat(image.convert('L')).stddev[0] < 8:
            return None
        vector = encode([image], 'image')[0]
        products, vectors = self.visual_index
        scores = vectors @ vector
        order = np.argsort(-scores, kind='stable')
        if len(order):
            best = scores[order[0]]
            runner_up = scores[order[1]] if len(order) > 1 else -np.inf
            if best >= 0.93 or (best >= 0.82 and best - runner_up >= 0.035):
                return products[order[0]]
        return None
