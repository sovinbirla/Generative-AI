"""Streamlit storefront chat demonstration."""
from io import BytesIO
import re

from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from catalog import DATA, image_path
from grounded import grounded_answer
from rag import INDEX, MODES, Store, answer
from retrieval import CatalogLookup, REFERENCES

st.set_page_config(page_title='Catalog Assist', page_icon='🛍️')
st.title('Catalog Assist')
st.caption('Search the January 2020 Amazon catalog and verified product references. Amazon catalog prices are historical.')


@st.cache_resource
def get_lookup(index_version, catalog_version, reference_version):
    return CatalogLookup(Store())


if not INDEX.exists():
    st.error('Build the catalog index with python rag.py index before starting the app.')
    st.stop()

catalog_file = DATA / 'amazon.csv'
lookup = get_lookup(INDEX.stat().st_mtime_ns,
                    catalog_file.stat().st_mtime_ns if catalog_file.is_file() else None,
                    REFERENCES.stat().st_mtime_ns)
with st.sidebar:
    mode = st.selectbox('Prompt examples', list(MODES), index=1,
                        format_func={'zero': 'Zero-shot', 'few': 'Few-shot (2)', 'multi': 'Multi-shot (4)'}.get)
    if st.button('Clear conversation', key='clear_chat'):
        st.session_state.turns = []
        st.session_state.focus = []
    st.caption(f'{lookup.historical_count:,} Amazon products searchable by name')
    st.caption(f'{len(lookup.references)} verified product references')

turns = st.session_state.setdefault('turns', [])
st.session_state.setdefault('focus', [])


def show(turn):
    with st.chat_message('user'):
        st.write(turn['question'])
        if turn.get('upload'):
            st.image(turn['upload'], width=180)
    with st.chat_message('assistant'):
        st.write(turn['response'])
        for i, product in enumerate(turn['products'], 1):
            with st.expander(f"[P{i}] {product['title']}", expanded=turn['show_images'] and i == 1):
                if image_path(product).exists():
                    st.image(str(image_path(product)), width=230)
                if product.get('source_name'):
                    st.caption('Verified product reference')
                if product['price']:
                    st.write(f"Catalog price {product['price']}")
                st.write(product['features'])
                if product.get('source_name'):
                    st.link_button(product['source_name'], product['url'])
                    if product.get('reference_photo_url'):
                        st.link_button('Verify reference photo', product['reference_photo_url'])
                elif product['url'].startswith('https://www.amazon.com/'):
                    st.link_button('Open original listing', product['url'])


conversation = st.container()
with conversation:
    for turn in turns:
        show(turn)

with st.form('message', clear_on_submit=True):
    question = st.text_input('Product question')
    upload = st.file_uploader('Optional product photo', type=['jpg', 'jpeg', 'png', 'webp'])
    follow_up = st.checkbox('Use previous product matches for this follow-up')
    submitted = st.form_submit_button('Send', type='primary')

if submitted:
    upload_preview = None
    photo = None
    if upload:
        try:
            with Image.open(upload) as source:
                if source.width * source.height > 25_000_000:
                    raise ValueError('Image dimensions exceed limit.')
                photo = ImageOps.exif_transpose(source).convert('RGB')
                photo.thumbnail((768, 768))
            buffer = BytesIO()
            photo.save(buffer, format='JPEG')
            upload_preview = buffer.getvalue()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            st.error('Upload a readable image up to 25 megapixels.')
            st.stop()
    question = question.strip()
    if not (question or photo):
        st.error('Enter a product question or upload a photo.')
        st.stop()
    prompt = question or 'Identify the product and describe what it is used for.'
    with st.spinner('Searching the catalog and composing an answer…'):
        try:
            comparison = re.fullmatch(
                r'(?:can you )?compare\s+(.+?)\s+(?:with|versus|vs\.?|and)\s+(.+?)\??',
                question, flags=re.I,
            )
            if follow_up and photo is None and st.session_state.focus:
                products = st.session_state.focus
            elif comparison and photo is None:
                matches = [lookup.find_text(name) for name in comparison.groups()]
                products = [hit[0] for hit in matches] if all(matches) else []
            elif photo is not None:
                match = lookup.find_photo(photo)
                products = [match] if match is not None else []
            else:
                products = lookup.find_text(question)

            if not products:
                if photo is not None:
                    response = ('I couldn’t confidently match this photo to a product. '
                                'Try a clear photo of the product label or search by brand and model.')
                else:
                    response = ('I couldn’t find a supported match for the product in the available Amazon '
                                'catalog or verified references. Check the product name or model number.')
            else:
                response = grounded_answer(prompt, products, image=photo is not None)
                if response is None:
                    response = answer(prompt, products, mode, [t['question'] for t in turns])
        except (OSError, RuntimeError) as error:
            st.error(f'Generation failed: {error}')
            st.stop()
    st.session_state.focus = products
    turn = dict(question=prompt, response=response, products=products, upload=upload_preview,
                show_images=any(word in question.lower() for word in ('picture', 'photo', 'image')))
    turns.append(turn)
    with conversation:
        show(turn)
