#!/usr/bin/env python3
"""Week 5 solutions: reproducible arithmetic, PyTorch demonstrations, and PDF.

Run from the course workspace:
    .venv/bin/python 'AI Strategy Assignment/Week5_Solutions.py'
    .venv/bin/python 'AI Strategy Assignment/Week5_Solutions.py' --build-pdf
    .venv/bin/python 'AI Strategy Assignment/Week5_Solutions.py' --pdf-only

Dependencies: torch for demonstrations; reportlab and markdown-it-py for PDF.
The neighboring Week5_Homework_Solutions.md is the editable PDF source.
No downloads, external API calls, or real transaction/maintenance data are used.
The demonstrations establish implementation behavior, not predictive performance.
Source notebook cells are numbered from zero in the raw .ipynb cell array.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def challenge_math() -> dict:
    """Apply the supplied lecture's decimal-GB, Chinchilla, and 6ND rules."""
    clean_bytes = 10 * 10**9
    tokens = clean_bytes / 4
    required = 20 * 3 * 10**9
    flops = 6 * 8 * 10**9 * tokens
    hours = flops / (4 * 10**14) / 3600
    assert math.isclose(tokens, 2.5e9)
    assert math.isclose(required / tokens, 24)
    assert math.isclose(hours, 83.33333333333333)
    return {
        "assumption": "10 decimal GB of clean English text; 4 bytes/token",
        "tokens": tokens,
        "3B_scratch_tokens_at_20_per_parameter": required,
        "data_shortfall_factor": required / tokens,
        "data_fraction_of_rule_of_thumb": tokens / required,
        "8B_CPT_FLOPs_one_epoch": flops,
        "aggregate_H100_GPU_hours_one_epoch": hours,
        "mixed_precision_Adam_8B_GB_before_activations": 16 * 8e9 / 1e9,
        "note": "Aggregate GPU-hours, not a promise of single-GPU feasibility or wall time.",
    }


def storage_math() -> dict:
    """Illustrative upper scenario, not a measurement of the supplied manuals."""
    # 500-token windows with 100-token overlap have a 400-token stride.
    chunks = math.ceil((2.5e9 - 500) / 400) + 1
    return {
        "1B_4bit_weights_GB_excluding_overhead": 1e9 * 4 / 8 / 1e9,
        "3B_4bit_weights_GB_excluding_overhead": 3e9 * 4 / 8 / 1e9,
        "assumed_clean_corpus_tokens": 2.5e9,
        "window_tokens": 500,
        "overlap_tokens": 100,
        "estimated_chunks": chunks,
        "384D_FP16_vectors_GB_excluding_index_overhead": chunks * 384 * 2 / 1e9,
        "note": "Raw files may yield much less text; measure actual storage and peak RAM.",
    }


def neural_demos() -> dict:
    """Q1 architectures and Q2 fixed-architecture learning, using synthetic data."""
    import torch
    from torch import nn

    torch.manual_seed(32027)
    torch.set_num_threads(1)

    class StructuredPreprocessor:
        """Fit medians, means, and population stds on training data only.

        Expects nonnegative amount and account age. NaNs are imputed; infinite
        values and negative observations are rejected. Missingness flags could
        be appended in a production model, changing the 770D input to 772D.
        """

        def _transform(self, x):
            if x.ndim != 2 or x.shape[1] != 2:
                raise ValueError("Expected shape (batch, 2).")
            if torch.isinf(x).any() or (x[~torch.isnan(x)] < 0).any():
                raise ValueError("Use nonnegative finite values or NaN.")
            return torch.log1p(x)

        def fit(self, x):
            z = self._transform(x)
            self.median = torch.nanmedian(z, dim=0).values
            if torch.isnan(self.median).any():
                raise ValueError("Each feature needs an observed training value.")
            z = torch.where(torch.isnan(z), self.median, z)
            self.mean = z.mean(dim=0)
            self.std = z.std(dim=0, unbiased=False).clamp_min(1e-6)
            return self

        def transform(self, x):
            z = self._transform(x)
            z = torch.where(torch.isnan(z), self.median, z)
            return (z - self.mean) / self.std

    class EarlyFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_norm = nn.LayerNorm(768)
            self.head = nn.Sequential(nn.Linear(770, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, embedding, structured):
            return self.head(torch.cat([self.text_norm(embedding), structured], dim=-1))

    class TwoBranchFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.text = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 64), nn.ReLU())
            self.structured = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(80, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, embedding, structured):
            return self.head(torch.cat([self.text(embedding), self.structured(structured)], dim=-1))

    train_s = torch.tensor([[10., 1.], [100., 30.], [1000., 365.], [50., 90.]])
    prep = StructuredPreprocessor().fit(train_s)
    s = prep.transform(train_s)
    assert torch.allclose(s.mean(0), torch.zeros(2), atol=1e-6)
    assert torch.isfinite(prep.transform(torch.tensor([[float("nan"), 10.]]))).all()
    e = torch.randn(4, 768)  # Stand-in vectors, not embeddings from actual text.
    labels = torch.tensor([[0.], [1.], [1.], [0.]])
    architectures = {}
    for cls in (EarlyFusion, TwoBranchFusion):
        model = cls()
        logits = model(e, s)
        assert logits.shape == (4, 1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        architectures[cls.__name__] = {
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "output_shape": list(logits.shape),
            "finite_backward_pass": True,
        }

    # A constructive counterexample: 768 text coefficients can all be zero.
    # The two structured coefficients alone can then determine the output.
    w_text = torch.zeros(768)
    w_s = torch.tensor([3., -2.])
    fixed_text = torch.randn(768)
    p0 = torch.sigmoid(fixed_text @ w_text + torch.tensor([0., 0.]) @ w_s).item()
    p1 = torch.sigmoid(fixed_text @ w_text + torch.tensor([1., 0.]) @ w_s).item()
    assert math.isclose(p0, 0.5) and p1 > 0.95

    class TinyAttention(nn.Module):
        """Raschka Ch. 3-style Q/K/V projections: 3 * (3 * 2) = 18 weights.

        Uses sqrt(key dimension) scaling. This is a toy MSE demonstration,
        not a simulation of CPT, IFT, CFT, or a maintenance model.
        """
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(3, 2, bias=False)
            self.k = nn.Linear(3, 2, bias=False)
            self.v = nn.Linear(3, 2, bias=False)

        def forward(self, x):
            q, k, v = self.q(x), self.k(x), self.v(x)
            scores = q @ k.T / math.sqrt(k.shape[-1])
            return scores.softmax(dim=-1) @ v

    # Fixed inputs from Self_Attention_Sebastian_CH3.ipynb, raw cell 24.
    x = torch.tensor([[.43, .15, .89], [.55, .87, .66], [.57, .85, .64],
                      [.22, .58, .33], [.77, .25, .10], [.05, .80, .55]])
    model = TinyAttention()
    before_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    shapes_before = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    before = model(x).detach().clone()
    loss_before = before.square().mean().item()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
    after = model(x).detach()
    loss_after = after.square().mean().item()
    shapes_after = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert shapes_before == shapes_after
    assert sum(p.numel() for p in model.parameters()) == 18
    assert not torch.allclose(before, after)
    assert loss_after < loss_before
    delta = max((model.state_dict()[k] - v).abs().max().item() for k, v in before_weights.items())
    return {
        "Q1_architectures": architectures,
        "Q1_dimension_counterexample": {"p_at_amount_z0": p0, "p_at_amount_z1": p1},
        "Q2_learning_same_architecture": {
            "parameter_count_before_and_after": 18,
            "all_parameter_shapes_unchanged": shapes_before == shapes_after,
            "loss_before": loss_before, "loss_after": loss_after,
            "largest_parameter_change": delta,
            "outputs_changed": True,
        },
        "limitation": "Synthetic functional demonstrations; no real fraud or aviation accuracy measured.",
    }


def build_pdf(path: Path) -> None:
    """Compile the neighboring Markdown source, including its layout hints."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak,
    )

    mono_font = "Courier"
    for font_path in (
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ):
        if font_path.is_file():
            pdfmetrics.registerFont(TTFont("Week5Mono", str(font_path)))
            mono_font = "Week5Mono"
            break

    navy = colors.HexColor("#17334A")
    teal = colors.HexColor("#147D86")
    ink = colors.HexColor("#243541")
    muted = colors.HexColor("#5A6E7D")
    pale = colors.HexColor("#EFF5F7")
    line = colors.HexColor("#D3E0E6")
    width = 504
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleX", fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=navy, spaceAfter=10))
    styles.add(ParagraphStyle("SubX", fontName="Helvetica", fontSize=10, leading=14, textColor=muted, spaceAfter=12))
    styles.add(ParagraphStyle("HeadX", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=teal, spaceBefore=11, spaceAfter=5))
    styles.add(ParagraphStyle("BodyX", fontName="Helvetica", fontSize=10, leading=13.5, textColor=ink, spaceAfter=8))
    styles.add(ParagraphStyle("SmallX", fontName="Helvetica", fontSize=8.5, leading=11.2, textColor=ink, spaceAfter=6))
    styles.add(ParagraphStyle("TinyX", fontName="Helvetica", fontSize=7.8, leading=10.2, textColor=muted, spaceAfter=5))
    styles.add(ParagraphStyle("CellX", fontName="Helvetica", fontSize=8.6, leading=11.2, textColor=ink))
    styles.add(ParagraphStyle("CellHeadX", fontName="Helvetica-Bold", fontSize=8.6, leading=11.2, textColor=colors.white))
    styles.add(ParagraphStyle("EquationX", fontName=mono_font, fontSize=9.3, leading=13, textColor=navy, backColor=pale, borderPadding=9, spaceBefore=3, spaceAfter=12))
    story = []

    def p(text, style="BodyX"):
        return Paragraph(text, styles[style])

    def add(text, style="BodyX"):
        story.append(p(text, style))

    def table(rows, widths):
        data = [[p(t, "CellHeadX" if i == 0 else "CellX") for t in row] for i, row in enumerate(rows)]
        t = Table(data, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, -1), (-1, -1), .5, line),
        ]))
        story.append(t)

    from html import escape
    from markdown_it import MarkdownIt
    from reportlab.platypus import Preformatted

    markdown_path = Path(__file__).resolve().with_name("Week5_Homework_Solutions.md")
    markdown = MarkdownIt("commonmark", {"html": True}).enable("table")
    tokens = markdown.parse(markdown_path.read_text(encoding="utf-8"))

    def inline(token):
        chunks = []
        tags = {"strong_open": "<b>", "strong_close": "</b>",
                "em_open": "<i>", "em_close": "</i>"}
        for child in token.children or []:
            if child.type == "text":
                chunks.append(escape(child.content))
            elif child.type in tags:
                chunks.append(tags[child.type])
            elif child.type == "code_inline":
                chunks.append(f'<font name="{mono_font}">' + escape(child.content) + '</font>')
            elif child.type == "html_inline":
                chunks.append(child.content.replace("<sup>", "<super>").replace("</sup>", "</super>"))
            elif child.type in ("softbreak", "hardbreak"):
                chunks.append(" " if child.type == "softbreak" else "<br/>")
            else:
                raise ValueError(f"Unsupported Markdown inline token: {child.type}")
        return "".join(chunks)

    next_style = "BodyX"
    next_widths = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "html_block":
            directive = token.content.strip()
            if directive == "<!-- pagebreak -->":
                story.append(PageBreak())
            elif directive.startswith("<!-- style:"):
                next_style = directive.removeprefix("<!-- style:").removesuffix("-->").strip()
                if next_style not in styles:
                    raise ValueError(f"Unknown paragraph style: {next_style}")
            elif directive.startswith("<!-- widths:"):
                next_widths = [float(x) for x in directive.removeprefix("<!-- widths:").removesuffix("-->").split(",")]
        elif token.type == "heading_open":
            add(inline(tokens[i + 1]), "TitleX" if token.tag == "h1" else "HeadX")
            i += 2
        elif token.type == "paragraph_open":
            add(inline(tokens[i + 1]), next_style)
            next_style = "BodyX"
            i += 2
        elif token.type == "fence":
            if token.info.strip() == "equation":
                add(escape(token.content.strip()), "EquationX")
            else:
                diagram_style = ParagraphStyle("DiagramText", fontName=mono_font, fontSize=8,
                                               leading=11, textColor=navy, spaceAfter=12)
                story.append(Preformatted(token.content.rstrip(), diagram_style))
        elif token.type == "table_open":
            rows = []
            i += 1
            while tokens[i].type != "table_close":
                if tokens[i].type == "tr_open":
                    rows.append([])
                elif tokens[i].type == "inline":
                    rows[-1].append(inline(tokens[i]))
                i += 1
            widths = next_widths or [width / len(rows[0])] * len(rows[0])
            if len(widths) != len(rows[0]) or abs(sum(widths) - width) > .1:
                raise ValueError("Table widths must match the columns and sum to 504 points.")
            table(rows, widths)
            next_widths = None
        else:
            raise ValueError(f"Unsupported Markdown block: {token.type}")
        i += 1

    def footer(c, doc):
        c.saveState()
        c.setStrokeColor(line)
        c.line(54, 43, 558, 43)
        c.setFont("Helvetica", 8)
        c.setFillColor(muted)
        c.drawString(54, 29, "ADSP 32027 | Generative AI | Week 5 solutions")
        c.drawRightString(558, 29, str(doc.page))
        c.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=(612, 792), rightMargin=54,
                            leftMargin=54, topMargin=43, bottomMargin=57,
                            title="Week 5 Homework Solutions - Domain Models",
                            author="Coursework solution", subject="ADSP 32027 Week 5")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-pdf", action="store_true", help="Regenerate the PDF beside this script.")
    parser.add_argument("--pdf-only", action="store_true", help="Compile Markdown to PDF without PyTorch demonstrations.")
    args = parser.parse_args()
    if args.pdf_only:
        destination = Path(__file__).resolve().with_name("Week5_Homework_Solutions.pdf")
        build_pdf(destination)
        print(f"Created: {destination}")
        return
    results = {"challenge": challenge_math(), "offline_storage": storage_math(), "demonstrations": neural_demos()}
    print(json.dumps(results, indent=2))
    if args.build_pdf:
        destination = Path(__file__).resolve().with_name("Week5_Homework_Solutions.pdf")
        build_pdf(destination)
        print(f"\nCreated: {destination}")


if __name__ == "__main__":
    main()
