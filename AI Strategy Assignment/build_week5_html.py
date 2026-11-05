#!/usr/bin/env python3
"""Build a standalone HTML edition of the Week 5 Markdown solutions.

From the course workspace:
    .venv/bin/python 'AI Strategy Assignment/build_week5_html.py'

Requires markdown-it-py. All styling and diagrams are embedded in the HTML.
"""

from html import escape
from pathlib import Path
import re

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parent


def box(x, y, w, h, title, lines=(), kind="neutral"):
    palette = {
        "text": ("#edf4ff", "#b8cff3", "#2459a5"),
        "data": ("#fff4e9", "#edc99e", "#914d16"),
        "fusion": ("#e9f6f3", "#a4d5c9", "#126456"),
        "neutral": ("#ffffff", "#cdd8e1", "#213c51"),
        "output": ("#18384d", "#18384d", "#ffffff"),
    }
    fill, stroke, color = palette[kind]
    result = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}"/>'
    text_y = y + (h - (len(lines) * 21 + 21)) / 2 + 17
    result += f'<text x="{x+w/2}" y="{text_y}" text-anchor="middle" fill="{color}" font-size="16" font-weight="700">{escape(title)}</text>'
    for n, line in enumerate(lines):
        result += f'<text x="{x+w/2}" y="{text_y+24+n*21}" text-anchor="middle" fill="{color}" font-size="14">{escape(line)}</text>'
    return result


def arrow(path, dashed=False):
    return f'<path d="{path}" fill="none" stroke="#6b8598" stroke-width="2" stroke-linejoin="round" marker-end="url(#arrow)"' + (' stroke-dasharray="6 5"' if dashed else '') + '/>'


def svg_start(name, title, desc, height):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 {height}" role="img" aria-labelledby="{name}-title {name}-desc" style="font-family:Arial,Helvetica,sans-serif">
<title id="{name}-title">{escape(title)}</title>
<desc id="{name}-desc">{escape(desc)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6b8598"/></marker></defs>
<rect width="1120" height="{height}" rx="16" fill="#ffffff"/>
'''


def fusion_diagram():
    svg = svg_start(
        "fusion", "Two architectures for text and structured features",
        "A: normalize a 768-dimensional text embedding and transform two numeric features, concatenate to 770, then train a 64-unit MLP and sigmoid. B: project text to 64 dimensions and numeric features to 16, concatenate to 80, then train an 80-to-32-to-1 fusion head and sigmoid. Both use training-set preprocessing and fraud labels.",
        730,
    )
    for top, letter, title, subtitle in (
        (18, "A", "Early concatenation", "Baseline: combine features before the learned classifier"),
        (350, "B", "Separate branches, then fusion", "Recommended candidate: learn a representation for each input type"),
    ):
        svg += f'<rect x="18" y="{top}" width="1084" height="312" rx="16" fill="#f8fafc" stroke="#dce5ec"/>'
        svg += f'<circle cx="52" cy="{top+34}" r="17" fill="#18384d"/><text x="52" y="{top+40}" text-anchor="middle" fill="white" font-size="17" font-weight="700">{letter}</text>'
        svg += f'<text x="82" y="{top+32}" fill="#18384d" font-size="20" font-weight="700">{title}</text><text x="82" y="{top+55}" fill="#526c7d" font-size="14">{subtitle}</text>'
        svg += box(42, top+88, 172, 78, "Text embedding", ["768 dimensions"], "text")
        svg += box(42, top+210, 172, 78, "Amount + age", ["2 numeric features"], "data")
        if letter == "A":
            svg += box(248, top+88, 210, 78, "Normalize text", ["LayerNorm"], "text")
            svg += box(248, top+210, 210, 78, "Transform numbers", ["log1p + train-set scaling"], "data")
            dim, head = "770 = 768 + 2", ["770 → 64 → ReLU → 1"]
        else:
            svg += box(248, top+88, 210, 78, "Text branch → 64", ["LayerNorm → Linear → ReLU"], "text")
            svg += box(248, top+210, 210, 78, "Numeric branch → 16", ["log1p + train-set scaling", "Linear → ReLU"], "data")
            dim, head = "80 = 64 + 16", ["80 → 32 → ReLU → 1"]
        svg += box(508, top+148, 146, 82, "Concatenate", [dim], "fusion")
        svg += box(694, top+148, 172, 82, "Learned MLP", head)
        svg += box(906, top+148, 172, 82, "Fraud probability", ["1 logit → sigmoid", "P(fraud | inputs)"], "output")
        for path in (
            f'M214 {top+127} H248', f'M214 {top+249} H248',
            f'M458 {top+127} H482 V{top+173} H508',
            f'M458 {top+249} H482 V{top+206} H508',
            f'M654 {top+189} H694', f'M866 {top+189} H906',
        ):
            svg += arrow(path)
    svg += '<text x="34" y="696" fill="#526c7d" font-size="15">Training: fraud labels + binary cross-entropy on logits. Apply sigmoid at inference.</text>'
    svg += '<text x="34" y="720" fill="#526c7d" font-size="15">Fit preprocessing on training data only. More dimensions do not automatically mean more influence.</text></svg>'
    return svg


def offline_diagram():
    svg = svg_start("offline", "Offline maintenance assistant architecture", "Central preparation produces a signed package of models, approved manuals and retrieval indexes. On the iPad, aircraft applicability and package validity checks precede local retrieval, a small generator, evidence checks and cited answers. Unsupported or expired advice is withheld.", 930)
    svg += '<rect x="18" y="18" width="1084" height="340" rx="16" fill="#f8fafc" stroke="#dce5ec"/><text x="42" y="54" fill="#18384d" font-size="19" font-weight="700">01 / CENTRAL PREPARATION</text>'
    svg += box(42, 88, 286, 100, "Approved 10 GB corpus", ["OCR, tables, revisions", "Aircraft applicability"], "text")
    svg += box(416, 88, 286, 100, "Prepare retrieval", ["Chunks + BM25 + embeddings", "Source-page links"], "fusion")
    svg += box(790, 88, 286, 100, "Prepare local models", ["Small generator + encoder", "Optional IFT; 4-bit export"])
    svg += box(300, 244, 520, 84, "Validated, signed package", ["Models + manuals + index + version / expiry manifest"], "output")
    for p in ("M328 138 H416", "M559 188 V216 H470 V244", "M933 188 V216 H650 V244"):
        svg += arrow(p)
    svg += '<path d="M18 389 H1102" stroke="#b0bdc8" stroke-dasharray="6 6"/><rect x="344" y="373" width="432" height="32" fill="white"/><text x="560" y="394" text-anchor="middle" fill="#526c7d" font-size="14">Install the package while connected</text>'
    svg += '<text x="42" y="438" fill="#126456" font-size="19" font-weight="700">02 / ON THE IPAD · ALL INFERENCE IS OFFLINE</text>'
    svg += box(42, 470, 286, 96, "Technician’s question", ["Aircraft / tail, task", "Observed symptoms"], "text")
    svg += box(416, 470, 286, 96, "Applicability + validity", ["Package expiry; fleet / tail", "Approved revision filters"], "fusion")
    svg += box(790, 470, 286, 96, "Local hybrid retrieval", ["BM25 + small query encoder", "Select relevant evidence"], "fusion")
    svg += box(790, 636, 286, 104, "Local retrieval store", ["Chunks + metadata + index", "Approved original pages"])
    svg += box(416, 636, 286, 104, "Small local generator", ["Question + source excerpts", "Grounded draft; short context"])
    svg += box(42, 636, 286, 104, "Evidence checks", ["Verify sources and support", "Conflicts / gaps → abstain"], "fusion")
    svg += box(42, 806, 660, 90, "Answer + local source viewer", ["Document, revision, page / task links", "Expired or unsupported → withhold advice; Maintenance Control"], "output")
    for p in ("M328 518 H416", "M702 518 H790", "M933 636 V566", "M790 546 H746 V606 H559 V636", "M416 688 H328", "M185 740 V806"):
        svg += arrow(p)
    svg += arrow("M933 740 V851 H702", dashed=True)
    return (svg + '</svg>').replace('id="arrow"', 'id="offline-arrow"').replace('url(#arrow)', 'url(#offline-arrow)')


CSS = r"""
:root{--navy:#18384d;--teal:#126456;--ink:#233d4e;--muted:#607482;--line:#dbe4e9;--paper:#fff;--bg:#f3f5f6}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:30px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}a{color:var(--teal);text-underline-offset:3px}button{font:inherit;cursor:pointer}.wrap{max-width:1460px;margin:auto;padding:44px 44px 80px}.hero{background:var(--navy);color:white;padding:46px 50px;border-radius:20px;margin-bottom:28px;position:relative}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.17em;text-transform:uppercase;opacity:.75}.hero h1{font-size:clamp(30px,4vw,48px);line-height:1.15;letter-spacing:-.035em;margin:14px 0 16px;max-width:850px}.hero p{max-width:760px;line-height:1.6;color:#d4e4eb;margin:0}.hero-meta{display:flex;flex-wrap:wrap;gap:10px;margin-top:24px}.pill{font-size:12px;border:1px solid #658091;border-radius:30px;padding:5px 13px;color:#e5eff3}.print{position:absolute;right:32px;top:28px;background:transparent;border:1px solid #78909e;border-radius:8px;color:#fff;padding:7px 13px;font-size:13px}.layout{display:grid;grid-template-columns:216px minmax(0,1fr);gap:30px;align-items:start}.toc{position:sticky;top:28px;padding:20px 0}.toc-label{font-size:11px;font-weight:800;letter-spacing:.15em;color:var(--muted);margin:0 12px 18px}.toc a{display:block;color:#536b7b;text-decoration:none;padding:10px 12px;border-left:2px solid #d5dee5;font-size:13px;line-height:1.5}.toc a:hover,.toc a:focus{color:var(--teal);border-color:var(--teal);background:#e8f1ef}.toc small{display:block;line-height:1.7;font-size:12px;color:var(--muted);margin:24px 12px}.solutions{min-width:0}.document-section{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:38px 42px;margin:0 0 24px;scroll-margin-top:24px}.section-number{display:block;font-size:11px;font-weight:800;letter-spacing:.14em;color:var(--teal);text-transform:uppercase;margin-bottom:9px}.document-section>h2{font-size:29px;letter-spacing:-.025em;line-height:1.22;margin:0 0 12px;color:var(--navy)}.document-section h3{font-size:18px;line-height:1.4;color:var(--teal);margin:30px 0 10px}.document-section p{margin:0 0 18px}.document-section p.subtitle{color:var(--muted);font-size:15px;padding-bottom:22px;border-bottom:1px solid var(--line);margin-bottom:24px}.document-section p.source-note{font-size:12px;color:var(--muted);line-height:1.65}strong{font-weight:650;color:#17384b}.table-scroll{overflow-x:auto;margin:22px 0 26px}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.6}th{background:var(--navy);color:white;text-align:left;font-size:12px;font-weight:650}th,td{padding:13px 15px;vertical-align:top;border-bottom:1px solid var(--line)}tbody tr:nth-child(even){background:#f4f8fa}td:first-child{font-weight:600;min-width:100px}th:first-child{border-radius:7px 0 0 0}th:last-child{border-radius:0 7px 0 0}.equation{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;background:#eef4f7;border-left:3px solid #7d9bab;border-radius:0 8px 8px 0;padding:16px 20px;margin:18px 0 22px;color:var(--navy)}figure{margin:26px -14px 30px;padding:18px;border:1px solid var(--line);border-radius:14px;background:#fff;break-inside:avoid}figure .figure-title{font-size:16px;font-weight:700;color:var(--navy);margin:0 0 12px}figure svg{display:block;width:100%;height:auto}figcaption{font-size:12px;color:var(--muted);line-height:1.7;margin:14px 4px 0}.diagram-scroll{overflow-x:auto}.diagram-scroll svg{min-width:680px}.diagram-legend{display:flex;gap:20px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:0 4px 12px}.key:before{content:"";display:inline-block;width:9px;height:9px;border-radius:3px;background:#b8cff3;margin-right:7px}.key.data:before{background:#edc99e}.key.fusion:before{background:#a4d5c9}footer{font-size:12px;color:var(--muted);margin:32px 0 0;text-align:center}pre{white-space:pre-wrap;overflow-wrap:anywhere}sup,sub{font-size:.72em;line-height:0}
@media(min-width:1300px){.wrap{padding-left:52px;padding-right:52px}.document-section{padding:44px 48px}}
@media(max-width:980px){.wrap{padding:24px}.layout{grid-template-columns:1fr;gap:12px}.toc{position:static;display:flex;gap:8px;overflow-x:auto;padding:0 0 12px}.toc-label,.toc small{display:none}.toc a{flex-shrink:0;border:1px solid var(--line);border-radius:7px;padding:8px 12px;background:white;font-size:12px}.hero{padding:38px 30px}.document-section{padding:30px}.print{position:static;margin-top:22px}}
@media(max-width:540px){body{font-size:15px}.wrap{padding:14px}.hero{padding:28px 24px;border-radius:14px}.document-section{padding:26px 20px}.document-section>h2{font-size:25px}.document-section h3{font-size:17px}figure{margin-left:-8px;margin-right:-8px;padding:12px}.table-scroll table{min-width:600px}}
@page{size:letter;margin:16mm 14mm}
@media print{html{scroll-behavior:auto}body{background:white;color:#172f40;font-size:10pt;line-height:1.5;print-color-adjust:exact;-webkit-print-color-adjust:exact}.wrap{padding:0;max-width:none}.hero{padding:18px 22px;margin:0 0 20px;border-radius:8px}.hero h1{font-size:26pt}.hero p{font-size:10pt}.hero-meta{margin-top:12px}.pill{font-size:8pt}.toc,.print{display:none}.layout{display:block}.document-section{border:0;border-radius:0;padding:0;margin:0;break-before:page}.document-section:first-child{break-before:auto}.document-section>h2{font-size:22pt}.document-section h3{font-size:12pt;margin-top:17px;break-after:avoid}.document-section p{margin-bottom:10px;orphans:3;widows:3}.document-section p.subtitle{padding-bottom:12px;margin-bottom:14px}.section-number{font-size:8pt}.table-scroll{overflow:visible;margin:12px 0 17px}table{font-size:8.5pt}th,td{padding:7px 9px}tr,figure{break-inside:avoid}thead{display:table-header-group}.equation{font-size:9pt;padding:10px 14px}figure{margin:15px 0 18px;padding:10px}.diagram-scroll{overflow:visible}.diagram-scroll svg{min-width:0}.figure-title{font-size:11pt!important}.diagram-legend,figcaption{font-size:8pt}footer{font-size:8pt}}
"""


def build():
    source = (ROOT / "Week5_Homework_Solutions.md").read_text(encoding="utf-8")
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    fusion = fusion_diagram()
    offline = offline_diagram()
    default_fence = md.renderer.rules["fence"]

    def fence(tokens, idx, options, env):
        token = tokens[idx]
        if token.info.strip() == "equation":
            return '<pre class="equation">' + escape(token.content.strip()) + '</pre>\n'
        if "A. EARLY CONCATENATION" in token.content:
            return '<figure id="q1-diagram"><p class="figure-title">Text + structured data: two fusion designs</p><div class="diagram-legend"><span class="key">Text features</span><span class="key data">Structured features</span><span class="key fusion">Fusion</span></div><div class="diagram-scroll">' + fusion + '</div><figcaption>Figure 1. Both designs learn the contribution of the text and numeric features. The two-branch design learns modality-specific projections; held-out results determine whether its added flexibility helps.</figcaption></figure>\n'
        if "CENTRAL PREPARATION" in token.content:
            return '<figure id="q3-diagram"><p class="figure-title">Versioned knowledge, local inference</p><div class="diagram-scroll">' + offline + '</div><figcaption>Figure 2. Solid arrows show preparation and local processing. The dashed arrow opens the original local source page. Every inference component below the boundary runs on the iPad.</figcaption></figure>\n'
        return default_fence(tokens, idx, options, env)

    md.renderer.rules["fence"] = fence
    ids = ["challenge", "question-1", "question-1-validation", "question-2", "question-3", "question-3-justification"]
    labels = ["In-class challenge", "Q1 · Combine features", "Q1 · Influence & validation", "Q2 · Learning & parameters", "Q3 · Offline architecture", "Q3 · Design justification"]
    sections = re.split(r"<!--\s*pagebreak\s*-->", source)
    assert len(sections) == len(ids), "Expected the six Markdown solution sections."
    rendered = []
    for index, (section, section_id) in enumerate(zip(sections, ids)):
        # Convert source-only print hints to semantic HTML styling.
        section = re.sub(r"<!--\s*style:\s*(\w+)\s*-->\s*", lambda m: '<!-- ' + m[1] + ' -->\n\n', section)
        section = re.sub(r"<!--(?!\s*(?:SubX|TinyX|SmallX)\s*-->).*?-->", "", section, flags=re.S)
        html = md.render(section)
        html = re.sub(r"<!-- (SubX|TinyX|SmallX) -->\s*<p>", lambda m: '<p class="' + {"SubX": "subtitle", "TinyX": "source-note", "SmallX": "body-note"}[m[1]] + '">', html)
        # Each document section has an h2; Markdown subsections become h3.
        html = html.replace('<h2>', '<h3>').replace('</h2>', '</h3>')
        html = html.replace('<h1>', '<h2>').replace('</h1>', '</h2>')
        html = html.replace('<table>', '<div class="table-scroll"><table>').replace('</table>', '</table></div>')
        rendered.append(f'<section class="document-section" id="{section_id}" aria-labelledby="heading-{section_id}"><span class="section-number">Part {index+1:02d}</span>' + html.replace('<h2>', f'<h2 id="heading-{section_id}">', 1) + '</section>')

    nav = ''.join(f'<a href="#{i}">{label}</a>' for i, label in zip(ids, labels))
    document = '''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Week 5 homework solutions: feature fusion, model learning and an offline maintenance assistant."><title>Week 5 Homework Solutions | ADSP 32027</title><style>''' + CSS + '''</style></head>
<body><div class="wrap"><header class="hero"><div class="eyebrow">ADSP 32027 / Generative AI / Autumn 2026</div><h1>Week 5<br>Homework solutions</h1><p>Representations, model learning, and offline domain AI. The in-class challenge is followed by individual answers to Questions 1–3.</p><div class="hero-meta"><span class="pill">Text + structured features</span><span class="pill">Learning &amp; parameters</span><span class="pill">Offline domain AI</span></div><button type="button" class="print" onclick="window.print()">Print / Save PDF</button></header><div class="layout"><nav class="toc" aria-label="Solutions contents"><div class="toc-label">CONTENTS</div>''' + nav + '''<small>Diagrams are embedded.<br>This document works offline.</small></nav><main class="solutions">''' + '\n'.join(rendered) + '''</main></div><footer>ADSP 32027 · Week 5 homework solutions · Generated from the editable Markdown source</footer></div></body></html>'''
    destination = ROOT / "Week5_Homework_Solutions.html"
    destination.write_text(document, encoding="utf-8")
    (ROOT / "Week5_Q1_Feature_Fusion.svg").write_text(fusion, encoding="utf-8")
    print(f"Created: {destination}")
    print(f"Created: {ROOT / 'Week5_Q1_Feature_Fusion.svg'}")


if __name__ == "__main__":
    build()
