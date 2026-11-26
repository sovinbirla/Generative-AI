"""Streamlit storefront chat demonstration."""
from io import BytesIO
import re

from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from catalog import image_path
from rag import INDEX, MODES, Store, answer

st.set_page_config(page_title='Catalog Assist', page_icon='🛍️')
st.title('Catalog Assist')
st.caption('Ask about products in the January 2020 Amazon catalog sample. Prices are historical listing values.')


@st.cache_resource
def get_store(index_version):
    return Store()


if not INDEX.exists():
    st.error('Build the catalog index with python rag.py index before starting the app.')
    st.stop()

store = get_store(INDEX.stat().st_mtime_ns)
with st.sidebar:
    mode = st.selectbox('Prompt examples', list(MODES), index=1,
                        format_func={'zero': 'Zero-shot', 'few': 'Few-shot (2)', 'multi': 'Multi-shot (4)'}.get)
    if st.button('Clear conversation', key='clear_chat'):
        st.session_state.turns = []
        st.session_state.focus = []
    st.caption(f'{len(store.products)} indexed products')

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
                if product['price']:
                    st.write(f"Catalog price {product['price']}")
                st.write(product['features'])
                if product['url'].startswith('https://www.amazon.com/'):
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
                hits = [store.search(name, k=1)[0][0] for name in comparison.groups()]
                products = list({product['id']: product for product in hits}.values())
            else:
                products = [product for product, _ in store.search(question, photo, k=3)]
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
