"""Answer product questions that can be resolved directly from retrieved evidence."""

import re


PRICE = re.compile(r'\b(?:price|costs?)\b|\bhow much(?: is (?:it|this))?[?.! ]*$', re.I)
WARRANTY = re.compile(r'\b(?:warrant(?:y|ies)|guarantee)\b', re.I)
USAGE = re.compile(r'\b(?:usage|used for|how (?:to|do you|do i|can i) use)\b', re.I)
COMPARISON = re.compile(r'\b(?:compare|versus|vs|cheaper|cheapest)\b', re.I)
PICTURE = re.compile(r'\b(?:picture|photo|image)\b', re.I)


def grounded_answer(question, products, *, image=False):
    """Return an evidence-backed answer when no free-form generation is needed."""
    if not products:
        return None
    if len(products) > 1 and COMPARISON.search(question):
        if any(p.get('source_name') for p in products):
            return '\n\n'.join(
                f"**{p['title']}**. {p['features'] or p['description'] or 'No features are listed.'} [P{i}]"
                for i, p in enumerate(products, 1)
            )
        return None

    product = products[0]
    title = product['title']
    reference = product.get('source_name')
    evidence_text = product.get('features', '') or product.get('description', '')
    visual_match = f'The closest visual match is **{title}**. [P1]'
    visual_note = (
        product.get('visual_note')
        or 'Check the product label or model number to confirm the match.'
    )

    if PRICE.search(question):
        price = product.get('price')
        if price:
            price_answer = f'The Amazon catalog snapshot lists **{title}** at **{price}**. [P1]'
        elif reference:
            price_answer = f'The product reference for **{title}** does not include an Amazon price. [P1]'
        else:
            price_answer = f'The catalog snapshot has no listed price for **{title}**. [P1]'
        return f'{visual_match} {price_answer} {visual_note}' if image else price_answer

    if WARRANTY.search(question):
        warranty = product.get('warranty')
        warranty_evidence = ' '.join(product.get(key, '') for key in ('features', 'description'))
        if warranty:
            return f'{warranty} [P1]'
        if reference or not WARRANTY.search(warranty_evidence):
            return (
                f'The available evidence for **{title}** does not specify a warranty. [P1]'
            )

    if image:
        details = evidence_text.split('|', 1)[0].strip()
        return ' '.join(part for part in (visual_match, details, product.get('usage'), visual_note) if part)

    if reference:
        details = product.get('usage') if USAGE.search(question) else None
        details = details or evidence_text
        if not details:
            return f'The {reference} reference identifies **{title}**. [P1]'
        return f'**{title}**. {details} [P1]'

    if PICTURE.search(question):
        return f'**{title}** [P1]'

    return None
