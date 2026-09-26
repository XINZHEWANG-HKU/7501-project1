from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report-wangxinzhe"
BUILD = REPORT / "_build"
OUT = REPORT / "MP1_Report_WangXinzhe_u3036801008.docx"

NAVY = "17365D"
TEAL = "1B7F79"
ORANGE = "D97706"
RED = "B54A4A"
CHARCOAL = "30343B"
MID = "667085"
LIGHT = "E9EEF5"
PALE = "F5F7FA"
GRID = "D9D9D9"
WHITE = "FFFFFF"


def load_metrics(run: str) -> dict:
    return json.loads((ROOT / "runs" / run / "metrics.json").read_text())


initial = load_metrics("ablation-e0-baseline-s17")
long_baseline = load_metrics("official-baseline-s17")
plain = load_metrics("student-long-s17")
distill = load_metrics("student-distill-g1-s17")
cache = load_metrics("student-cache-only-s17")
both = load_metrics("student-cache-distill-s17")

runs = {
    "No distillation or cache": plain,
    "Cache only": cache,
    "Distillation only": distill,
    "Distillation and cache": both,
}


def value_at(d: dict, step: int) -> float:
    for row in d.get("validation_history", []):
        if row["step"] == step:
            return row["bpb"]
    if step == d["training_config"]["steps"]:
        return d["final_validation"]["bpb"]
    raise KeyError((step, d.get("selected_step")))


final_bpb = {name: value_at(data, 19200) for name, data in runs.items()}
A = final_bpb["No distillation or cache"]
B = final_bpb["Cache only"]
C = final_bpb["Distillation only"]
D = final_bpb["Distillation and cache"]


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_text_center(draw, xy, text, fnt, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=fnt, fill="#" + fill)


def make_ablation_chart(path: Path):
    width, height = 1600, 900
    img = Image.new("RGB", (width, height), "#" + WHITE)
    d = ImageDraw.Draw(img)
    left, top, right, bottom = 155, 95, 65, 165
    x0, y0 = left, height - bottom
    x1, y1 = width - right, top
    ymin, ymax = 1.42, 1.58
    title_f = font(42, True)
    label_f = font(26)
    small_f = font(23)
    value_f = font(27, True)
    d.text((left, 24), "Validation BPB at 19,200 updates", font=title_f, fill="#" + CHARCOAL)
    for tick in [1.42, 1.46, 1.50, 1.54, 1.58]:
        y = y0 - (tick - ymin) / (ymax - ymin) * (y0 - y1)
        d.line((x0, y, x1, y), fill="#" + GRID, width=2)
        text = f"{tick:.2f}"
        box = d.textbbox((0, 0), text, font=small_f)
        d.text((x0 - 20 - (box[2] - box[0]), y - 14), text, font=small_f, fill="#" + MID)
    d.line((x0, y1, x0, y0), fill="#" + CHARCOAL, width=3)
    d.line((x0, y0, x1, y0), fill="#" + CHARCOAL, width=3)
    labels = ["No mechanisms", "Cache only", "Distillation only", "Both mechanisms"]
    vals = [A, B, C, D]
    colors = [CHARCOAL, TEAL, ORANGE, NAVY]
    slot = (x1 - x0) / 4
    bar_w = slot * 0.58
    for i, (label, val, color) in enumerate(zip(labels, vals, colors)):
        cx = x0 + slot * (i + 0.5)
        y = y0 - (val - ymin) / (ymax - ymin) * (y0 - y1)
        d.rounded_rectangle((cx - bar_w / 2, y, cx + bar_w / 2, y0), radius=8, fill="#" + color)
        draw_text_center(d, (cx, y - 28), f"{val:.4f}", value_f, color)
        words = label.split()
        if len(words) > 2:
            line1 = " ".join(words[:2])
            line2 = " ".join(words[2:])
            draw_text_center(d, (cx, y0 + 37), line1, label_f, CHARCOAL)
            draw_text_center(d, (cx, y0 + 70), line2, label_f, CHARCOAL)
        else:
            draw_text_center(d, (cx, y0 + 48), label, label_f, CHARCOAL)
    d.text((x1 - 190, y1 + 22), "Lower is better", font=small_f, fill="#" + MID)
    img.save(path, dpi=(180, 180))


def make_curve_chart(path: Path):
    width, height = 1600, 900
    img = Image.new("RGB", (width, height), "#" + WHITE)
    d = ImageDraw.Draw(img)
    left, top, right, bottom = 150, 100, 65, 130
    x0, y0 = left, height - bottom
    x1, y1 = width - right, top
    xmin, xmax = 800, 19200
    ymin, ymax = 1.43, 1.82
    title_f = font(42, True)
    tick_f = font(22)
    legend_f = font(22)
    d.text((left, 24), "Validation BPB during training", font=title_f, fill="#" + CHARCOAL)
    for tick in [1.45, 1.55, 1.65, 1.75]:
        y = y0 - (tick - ymin) / (ymax - ymin) * (y0 - y1)
        d.line((x0, y, x1, y), fill="#" + GRID, width=2)
        d.text((58, y - 13), f"{tick:.2f}", font=tick_f, fill="#" + MID)
    for tick in [0, 4000, 8000, 12000, 16000, 19200]:
        x = x0 + (tick - 0) / 19200 * (x1 - x0)
        d.line((x, y0, x, y0 + 8), fill="#" + CHARCOAL, width=2)
        label = f"{tick // 1000}k" if tick else "0"
        draw_text_center(d, (x, y0 + 35), label, tick_f, MID)
    d.line((x0, y1, x0, y0), fill="#" + CHARCOAL, width=3)
    d.line((x0, y0, x1, y0), fill="#" + CHARCOAL, width=3)
    palette = {
        "No distillation or cache": CHARCOAL,
        "Cache only": TEAL,
        "Distillation only": ORANGE,
        "Distillation and cache": NAVY,
    }
    for name, data in runs.items():
        rows = data.get("validation_history", [])
        points = []
        for row in rows:
            step, bpb = row["step"], row["bpb"]
            if step >= xmin and ymin <= bpb <= ymax:
                x = x0 + step / xmax * (x1 - x0)
                y = y0 - (bpb - ymin) / (ymax - ymin) * (y0 - y1)
                points.append((x, y))
        if len(points) >= 2:
            d.line(points, fill="#" + palette[name], width=6, joint="curve")
            for x, y in points:
                d.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#" + palette[name])
    legend_x, legend_y = 790, 125
    legend_labels = [
        ("No mechanisms", CHARCOAL),
        ("Cache only", TEAL),
        ("Distillation only", ORANGE),
        ("Both mechanisms", NAVY),
    ]
    for i, (label, color) in enumerate(legend_labels):
        row, col = divmod(i, 2)
        x = legend_x + col * 355
        y = legend_y + row * 43
        d.line((x, y, x + 54, y), fill="#" + color, width=6)
        d.text((x + 68, y - 13), label, font=legend_f, fill="#" + CHARCOAL)
    d.text((730, height - 54), "Training updates", font=tick_f, fill="#" + MID)
    img.save(path, dpi=(180, 180))


def make_cache_diagram(path: Path):
    width, height = 1600, 610
    img = Image.new("RGB", (width, height), "#" + WHITE)
    d = ImageDraw.Draw(img)
    title_f = font(40, True)
    box_f = font(26, True)
    sub_f = font(22)
    d.text((70, 24), "Causal within-window neural cache", font=title_f, fill="#" + CHARCOAL)

    def box(x, y, w, h, title, subtitle, color):
        d.rounded_rectangle((x, y, x + w, y + h), radius=14, fill="#" + color, outline="#" + color, width=3)
        draw_text_center(d, (x + w / 2, y + 43), title, box_f, WHITE)
        draw_text_center(d, (x + w / 2, y + 82), subtitle, sub_f, WHITE)

    def arrow(xa, ya, xb, yb, color=CHARCOAL):
        d.line((xa, ya, xb, yb), fill="#" + color, width=5)
        angle = math.atan2(yb - ya, xb - xa)
        for off in (2.55, -2.55):
            d.line((xb, yb, xb + 18 * math.cos(angle + off), yb + 18 * math.sin(angle + off)), fill="#" + color, width=5)

    box(70, 220, 280, 110, "Transformer", "causal hidden states", NAVY)
    box(480, 115, 300, 110, "Language model", "softmax distribution", CHARCOAL)
    box(480, 335, 300, 110, "Neural cache", "earlier keys and tokens", TEAL)
    box(925, 225, 260, 110, "Learned gate", "sigmoid mixture weight", ORANGE)
    box(1320, 225, 210, 110, "Prediction", "normalized log p", NAVY)
    arrow(350, 275, 480, 170)
    arrow(350, 275, 480, 390)
    arrow(780, 170, 925, 255)
    arrow(780, 390, 925, 305)
    arrow(1185, 280, 1320, 280)
    d.text((473, 480), "Only keys j < i are visible and value j is token j + 1", font=sub_f, fill="#" + MID)
    d.text((1010, 390), "p = (1 - g) pLM + g pcache", font=sub_f, fill="#" + CHARCOAL)
    img.save(path, dpi=(180, 180))


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color=GRID, size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn("w:" + margin))
        if node is None:
            node = OxmlElement("w:" + margin)
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_repeat_table_header(row):
    repeat_table_header(row)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def make_table(doc, headers, rows, widths=None, aligns=None, font_size=8.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    table.alignment = 1
    table.allow_autofit = False
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for idx, text in enumerate(headers):
        cell = hdr.cells[idx]
        shade_cell(cell, NAVY)
        set_cell_border(cell)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(str(text))
        run.bold = True
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor(255, 255, 255)
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for idx, text in enumerate(row):
            cell = cells[idx]
            if ridx % 2:
                shade_cell(cell, PALE)
            set_cell_border(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = aligns[idx] if aligns else (WD_ALIGN_PARAGRAPH.LEFT if idx == 0 else WD_ALIGN_PARAGRAPH.CENTER)
            run = p.add_run(str(text))
            run.font.size = Pt(font_size)
            run.font.color.rgb = RGBColor(35, 39, 47)
    if widths:
        for row in table.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = Cm(width)
    return table


def set_run_font(run, name="Arial", size=None, bold=None, color=None, italic=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_body(doc, text, bold_lead=None, keep=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.08
    p.paragraph_format.keep_together = keep
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_run_font(r, size=9.6, bold=True, color=CHARCOAL)
        text = text[len(bold_lead):]
    r = p.add_run(text)
    set_run_font(r, size=9.6, color=CHARCOAL)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.04
    for run in p.runs:
        set_run_font(run, size=9.3, color=CHARCOAL)
    if not p.runs:
        r = p.add_run(text)
        set_run_font(r, size=9.3, color=CHARCOAL)
    else:
        p.runs[0].text = text
    return p


def add_code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.45)
    p.paragraph_format.right_indent = Cm(0.2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    set_run_font(r, name="Consolas", size=7.4, color=CHARCOAL)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = False
    r = p.add_run(text)
    set_run_font(r, size=8.2, italic=True, color=MID)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(4 if level == 1 else 3)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    set_run_font(r, size=14 if level == 1 else 11, bold=True, color="000000")
    return p


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Page ")
    set_run_font(run, size=8, color=MID)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def setup_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(1.55)
    sec.bottom_margin = Cm(1.55)
    sec.left_margin = Cm(1.75)
    sec.right_margin = Cm(1.75)
    sec.header_distance = Cm(0.65)
    sec.footer_distance = Cm(0.65)
    sec.different_first_page_header_footer = True
    add_page_number(sec.footer.paragraphs[0])
    sec.first_page_footer.paragraphs[0].text = ""

    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    normal.font.size = Pt(9.6)
    normal.font.color.rgb = RGBColor.from_string(CHARCOAL)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.08

    for name, size in (("Title", 23), ("Heading 1", 14), ("Heading 2", 11)):
        style = doc.styles[name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
    title_ppr = doc.styles["Title"].element.get_or_add_pPr()
    for border in title_ppr.findall(qn("w:pBdr")):
        title_ppr.remove(border)
    return doc


def page_break(doc):
    doc.add_page_break()


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    ablation_chart = BUILD / "ablation.png"
    curve_chart = BUILD / "curves.png"
    cache_diagram = BUILD / "cache.png"
    make_ablation_chart(ablation_chart)
    make_curve_chart(curve_chart)
    make_cache_diagram(cache_diagram)

    doc = setup_document()
    doc.core_properties.title = "Small Language Model Improvement with Distillation and Neural Cache"
    doc.core_properties.author = "WangXinzhe"
    doc.core_properties.subject = "DASE7506 MP1 report"
    doc.core_properties.keywords = "language model, distillation, neural cache, BPB"

    # Page 1
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(30)
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run("Small Language Model Improvement with Distillation and Neural Cache")
    set_run_font(r, size=23, bold=True, color="000000")
    for line, size, bold in [
        ("DASE7506 MP1 Small Language Model Challenge", 12, True),
        ("WangXinzhe", 11, False),
        ("Student ID u3036801008", 11, False),
        ("26 September 2026", 9.5, False),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(line)
        set_run_font(r, size=size, bold=bold, color=CHARCOAL)

    add_heading(doc, "Abstract", 1)
    add_body(doc, (
        "This report improves the supplied GPT baseline under the fixed WikiText 2 protocol. "
        "The final predictor uses a 6 layer rotary Transformer with RMSNorm and SwiGLU, then adds two "
        "core mechanisms: self-distillation from a model trained only on the supplied training split, "
        "and a strictly causal neural cache that is reset for every 256 token evaluation window. "
        "At 19,200 updates, the architecture without either mechanism obtains 1.55637 validation bits "
        "per byte. Distillation alone reaches 1.49356, cache alone reaches 1.49256, and their combination "
        "reaches 1.44406 validation BPB. After model selection was frozen, the same checkpoint obtained "
        "1.4615442689258094 BPB on the complete test split. The combined predictor remains inside all "
        "measured evaluation limits at 4.484 times baseline validation CPU time, 1.790 GiB peak RAM, and "
        "25.08 MiB of inference assets."
    ))
    add_heading(doc, "Main findings", 1)
    make_table(doc,
        ["Evidence", "Result", "Interpretation"],
        [
            ["Initial baseline", "2.07108 BPB", "Reference 1,200 update model"],
            ["Equal target architecture comparison", "1.69939 to 1.55637", "Architecture improves BPB by 0.14302"],
            ["Core mechanism ablation", "1.55637 to 1.44406", "Combined gain is 0.11231 BPB"],
            ["Frozen full-test result", "1.4615442689258094 BPB", "Value submitted to course website"],
            ["Resource compliance", "4.484x CPU  1.790 GiB  25.08 MiB", "All measured limits pass"],
        ],
        widths=[5.2, 4.2, 7.5],
        font_size=8.2,
    )
    add_body(doc, (
        "All model selection and mechanism comparisons use the validation split. The full-test result is "
        "reported only as the final frozen-checkpoint score and was not used to revise the method."
    ))

    # Page 2
    page_break(doc)
    add_heading(doc, "1 Task and contribution", 1)
    add_body(doc, (
        "The course fixes the data, BPE 2048 tokenizer, causal 256 token evaluation windows, and the "
        "bits per byte scoring rule. The objective is to reduce BPB while keeping CPU scoring time at or "
        "below five times the supplied baseline, peak evaluation RAM below 4 GiB, and uncompressed "
        "inference assets below 64 MiB. Training length and architecture may change, but all learning must "
        "use the supplied training text and model selection must use validation data."
    ))
    add_heading(doc, "Baseline and improved architecture", 2)
    make_table(doc,
        ["Component", "Supplied GPT", "Improved model"],
        [
            ["Blocks and width", "4 blocks  width 128", "6 blocks  width 288"],
            ["Attention", "4 heads", "6 heads with causal SDPA"],
            ["Position", "Learned absolute embedding", "Rotary position embedding"],
            ["Normalization", "LayerNorm", "RMSNorm"],
            ["Feed forward network", "4x GELU MLP", "SwiGLU with width 768"],
            ["Regularization", "None", "Residual dropout 0.2"],
            ["Initialization", "Standard deviation 0.02", "Depth scaled residual outputs"],
            ["Parameters", "1,088,256", "6,565,536 without cache"],
        ],
        widths=[4.6, 5.6, 6.7],
        font_size=8.2,
    )
    add_body(doc, (
        "The larger architecture allocates the available inference budget to representation capacity. "
        "RoPE expresses relative position directly in query and key rotations, RMSNorm removes the mean "
        "subtraction used by LayerNorm, and SwiGLU adds a learned gate inside the feed forward block. "
        "The output projection shares weights with the token embedding. Residual output matrices are "
        "initialized with standard deviation 0.02 divided by the square root of twice the depth, which "
        "reduces activation growth at initialization."
    ))
    add_heading(doc, "Measured architecture gain", 2)
    add_body(doc, (
        "A 19,200 update run of the supplied architecture gives 1.69939 BPB at the final step. With the "
        "same seed, batch size, context, optimizer family, number of updates, and 157,286,400 processed "
        "targets, the improved architecture without distillation or cache gives 1.55637 BPB. The 0.14302 "
        "BPB reduction isolates the benefit of the architecture and training recipe from additional "
        "training data exposure."
    ))

    # Page 3
    page_break(doc)
    add_heading(doc, "2 Core mechanisms", 1)
    add_heading(doc, "Self distillation", 2)
    add_body(doc, (
        "The teacher is the validation-selected checkpoint from the no-cache improved model at update "
        "18,400. It was trained from random initialization on the supplied training split with seed 17. "
        "During student training the teacher is fixed in evaluation mode and receives the same causal "
        "training prefix as the student. The student objective is L equals one minus alpha times hard "
        "label cross entropy plus alpha times T squared KL from the teacher distribution to the student "
        "distribution. The experiments use alpha 0.5 and temperature T 1.0."
    ))
    add_body(doc, (
        "The soft distribution contains information about plausible alternatives beyond the observed next "
        "token. This regularizes the student toward a smoother target while retaining a 0.5 weight on the "
        "ground-truth label. The teacher is used only during training, so distillation does not change the "
        "student's inference parameters, assets, or CPU scoring path."
    ))
    add_heading(doc, "Causal neural cache", 2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1)
    p.add_run().add_picture(str(cache_diagram), width=Inches(6.55))
    add_caption(doc, "Figure 1  The base distribution and within-window cache are combined by a learned gate")
    add_body(doc, (
        "The cache normalizes final hidden states and compares each query with earlier hidden keys. A "
        "strict lower triangular mask permits key j only when j is smaller than query position i. Key j "
        "stores token j plus 1; therefore the newest token retrieved for query i is token i, which is "
        "already in the observed prefix. Similarity weights are divided by a learned temperature bounded "
        "between 0.05 and 0.50 and accumulated into a vocabulary distribution."
    ))
    add_body(doc, (
        "A learned sigmoid gate mixes the language model and cache probabilities. Its bias starts at minus "
        "2.5 so early training relies mostly on the base model. The cache is local to one forward call and "
        "contains no persistent state. It resets between windows and examples, satisfying the course rule "
        "that evaluation windows are independent and no state crosses a boundary."
    ))

    # Page 4
    page_break(doc)
    add_heading(doc, "3 Experimental protocol", 1)
    add_heading(doc, "Data and scoring", 2)
    add_body(doc, (
        "All runs use the supplied WikiText 2 text and fixed BPE 2048 tokenizer. Training samples random "
        "257 token spans and predicts the final 256 tokens. Validation scoring covers 376,599 targets and "
        "1,148,007 UTF 8 bytes in independent causal windows. Final test scoring covers 428,405 targets and "
        "1,292,013 UTF 8 bytes. BPB is summed negative log base 2 probability divided by the raw byte count. "
        "Development decisions use validation only; the test split is used once for the submitted score."
    ))
    add_heading(doc, "Controlled settings", 2)
    make_table(doc,
        ["Setting", "Value"],
        [
            ["Random seed", "17"],
            ["Updates and batch size", "19,200 updates  batch 32"],
            ["Targets per run", "157,286,400"],
            ["Context and vocabulary", "256 tokens  2,048 tokens"],
            ["Optimizer", "AdamW  beta1 0.9  beta2 0.95  weight decay 0.1"],
            ["Learning rate", "0.001 peak  100 update warmup  cosine decay to 0.0001"],
            ["Gradient clipping", "Global norm 1.0"],
            ["Training precision", "FP32 on NVIDIA GeForce GTX 1660 Ti"],
            ["Distillation", "alpha 0.5  temperature 1.0"],
        ],
        widths=[5.3, 11.6],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
        font_size=8.2,
    )
    add_heading(doc, "Experiment matrix", 2)
    make_table(doc,
        ["Run", "Distillation", "Cache", "Selection step", "Best validation BPB"],
        [
            ["No mechanisms", "No", "No", "18,400", f"{plain['validation']['bpb']:.5f}"],
            ["Cache only", "No", "Yes", "16,000", f"{cache['validation']['bpb']:.5f}"],
            ["Distillation only", "Yes", "No", "19,200", f"{distill['validation']['bpb']:.5f}"],
            ["Both mechanisms", "Yes", "Yes", "19,200", f"{both['validation']['bpb']:.5f}"],
        ],
        widths=[5.5, 2.7, 2.0, 3.1, 3.6],
        font_size=8.0,
    )
    add_body(doc, (
        "The four runs differ only in the two core mechanisms. Main causal claims use the final update for "
        "all four runs, so every comparison has identical processed targets. Validation-selected results "
        "are reported separately because their selected steps differ."
    ))

    # Page 5
    page_break(doc)
    add_heading(doc, "4 Main results", 1)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(ablation_chart), width=Inches(6.65))
    add_caption(doc, "Figure 2  Fixed-step comparison with equal training targets")
    make_table(doc,
        ["Distillation", "Cache", "BPB at 19,200", "Gain from no mechanisms"],
        [
            ["No", "No", f"{A:.5f}", "reference"],
            ["No", "Yes", f"{B:.5f}", f"{A-B:.5f}"],
            ["Yes", "No", f"{C:.5f}", f"{A-C:.5f}"],
            ["Yes", "Yes", f"{D:.5f}", f"{A-D:.5f}"],
        ],
        widths=[4.0, 3.2, 4.5, 5.2],
        font_size=8.3,
    )
    add_body(doc, (
        "Both mechanisms produce similar standalone gains. Cache lowers BPB by 0.06380 without "
        "distillation, while distillation lowers it by 0.06281 without cache. Combining them lowers BPB by "
        "0.11231, a 7.22 percent reduction from the no-mechanism architecture. The final validation BPB of "
        "1.44406 is 0.62703 below the initial 1,200 update course baseline, although that larger comparison "
        "also includes more capacity and training."
    ))
    add_heading(doc, "Interaction between mechanisms", 2)
    add_body(doc, (
        "Cache adds 0.04950 BPB of improvement when distillation is already active, and distillation adds "
        "0.04851 when cache is already active. These conditional gains remain substantial but are smaller "
        "than the standalone gains. The positive 0.01430 BPB interaction term indicates diminishing returns: "
        "both methods improve the probability assigned to plausible continuations, so part of their benefit "
        "overlaps rather than adding independently."
    ))

    # Page 6
    page_break(doc)
    add_heading(doc, "5 Training dynamics and mechanism analysis", 1)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(curve_chart), width=Inches(6.65))
    add_caption(doc, "Figure 3  Validation trajectories for the four controlled runs")
    add_body(doc, (
        "The cache improves the model throughout training rather than only at the final checkpoint. At "
        "update 12,800, cache-only BPB is 1.49750 compared with 1.56318 for the no-mechanism run. At the "
        "same update, distillation-only BPB is 1.51013 and the combined model is 1.45706. The combined curve "
        "continues to improve through update 19,200, supporting selection of the final checkpoint."
    ))
    add_heading(doc, "Why the cache helps", 2)
    add_body(doc, (
        "WikiText contains repeated names, entities, and topical vocabulary inside a 256 token window. The "
        "parametric head must represent these patterns in fixed weights, while the cache can directly reuse "
        "a recent token after matching a similar hidden context. Because the gate is learned at every "
        "position, the model can suppress the cache when repetition is unhelpful."
    ))
    add_heading(doc, "Why distillation helps", 2)
    add_body(doc, (
        "A one-hot label treats every incorrect token equally. The teacher distribution distinguishes "
        "reasonable alternatives from implausible ones and stabilizes training with a smoother target. "
        "The teacher and student share the same architecture, so this is self-distillation rather than "
        "compression from a larger model. Its gain is therefore best interpreted as optimization and "
        "regularization rather than transfer of external knowledge."
    ))
    add_heading(doc, "Failure modes and tradeoffs", 2)
    add_body(doc, (
        "The cache can overemphasize accidental repetition and adds an O(L squared) similarity matrix plus "
        "a vocabulary scatter. Distillation nearly doubles training forward computation because the teacher "
        "must score every batch. The observed overlap between their gains also means that further generations "
        "of self-distillation may yield smaller improvements."
    ))

    # Page 7
    page_break(doc)
    add_heading(doc, "6 Efficiency and compliance", 1)
    add_heading(doc, "Evaluation resource limits", 2)
    baseline_cpu = initial["validation"]["seconds"]
    final_cpu = json.loads((ROOT / "runs/student-cache-distill-s17/validation_ram_check.json").read_text())["seconds"]
    cpu_ratio = final_cpu / baseline_cpu
    checkpoint_bytes = (ROOT / "runs/student-cache-distill-s17/checkpoint.pt").stat().st_size
    code_bytes = (ROOT / "student.py").stat().st_size
    assets_mib = (checkpoint_bytes + code_bytes) / 2**20
    make_table(doc,
        ["Constraint", "Measured result", "Course limit", "Status"],
        [
            ["CPU FP32 scoring time", f"{final_cpu:.2f} s  {cpu_ratio:.3f}x baseline", "at most 5x", "Pass"],
            ["Peak CPU evaluation RAM", "1.790 GiB", "at most 4 GiB", "Pass"],
            ["Inference assets", f"{assets_mib:.2f} MiB", "at most 64 MiB", "Pass"],
        ],
        widths=[5.0, 5.0, 3.6, 3.3],
        font_size=8.3,
    )
    add_body(doc, (
        "CPU time is measured on the same four-thread machine and split for both models: 8.686 seconds for "
        "the supplied baseline and 38.952 seconds for the final predictor. The resulting 4.484 times ratio "
        "leaves about 10.3 percent of the permitted timing budget. RAM and assets have wider margins. The "
        "timing result is the tightest constraint and should be reproduced on the submitted software stack. "
        "The later full-test reproduction took 43.159 seconds; it is reported separately and is not divided "
        "by the validation baseline time because the two splits contain different target counts."
    ))
    add_heading(doc, "Training and search cost", 2)
    make_table(doc,
        ["Run", "Direct train time", "Processed targets", "GPU peak reserved"],
        [
            ["Long supplied baseline", f"{long_baseline['train_seconds']/60:.1f} min", "157,286,400", f"{long_baseline['peak_reserved_gb']:.2f} GB"],
            ["No mechanisms teacher", f"{plain['train_seconds']/60:.1f} min", "157,286,400", f"{plain['peak_reserved_gb']:.2f} GB"],
            ["Cache only", f"{cache['train_seconds']/60:.1f} min", "157,286,400", f"{cache['peak_reserved_gb']:.2f} GB"],
            ["Distillation only", f"{distill['train_seconds']/60:.1f} min", "157,286,400", f"{distill['peak_reserved_gb']:.2f} GB"],
            ["Both mechanisms", f"{both['train_seconds']/60:.1f} min", "157,286,400", f"{both['peak_reserved_gb']:.2f} GB"],
        ],
        widths=[5.0, 4.0, 4.3, 3.6],
        font_size=8.1,
    )
    add_body(doc, (
        "The recorded runs account for 5.32 hours of direct training, including the 1,200 update initial "
        "CPU baseline. The two distilled runs reuse the same no-mechanism teacher, but reuse does not erase "
        "its 3,726.8 second training cost. For checkpoint ancestry, the final model therefore depends on "
        "the teacher's 157,286,400 targets plus its own 157,286,400 targets."
    ))
    add_heading(doc, "Data and causal compliance", 2)
    add_body(doc, (
        "The teacher and all students learn only from the supplied training text. Validation is used for "
        "checkpoint selection; no validation or test targets are cached. Predictions at position i use only "
        "the prefix through i, and all cache tensors are created inside the current forward call. The tokenizer, "
        "data files, evaluator, context length, and scoring rule remain unchanged."
    ))

    # Page 8
    page_break(doc)
    add_heading(doc, "7 Reproduction and conclusion", 1)
    add_heading(doc, "Training command (method record only)", 2)
    add_code(doc, (
        "python train.py --implementation student --config configs\\student_long_cache.json ^\n"
        "  --device cuda --precision fp32 --seed 17 --steps 19200 --batch-size 32 ^\n"
        "  --eval-every 800 --teacher-checkpoint runs\\student-long-s17\\checkpoint.pt ^\n"
        "  --distill-alpha 0.5 --distill-temperature 1.0 ^\n"
        "  --run-dir runs\\student-cache-distill-s17"
    ))
    add_body(doc, "This command records training provenance; it is not needed to reproduce the submitted score.")
    add_heading(doc, "Frozen checkpoint test reproduction", 2)
    add_code(doc, (
        "python evaluate.py ^\n"
        "  --checkpoint runs\\student-cache-distill-s17\\checkpoint.pt ^\n"
        "  --device cpu --precision fp32 --threads 4 ^\n"
        "  --split test --output reproduced_test_cpu_fp32.json"
    ))
    add_code(doc, "python -m unittest discover -s tests -v")
    add_body(doc, (
        "This no-training command produced test BPB 1.4615442689258094, token perplexity 21.226944720949717, "
        "428,405 scored targets, and 1,292,013 UTF 8 bytes. The value submitted to the course website is the "
        "JSON bpb field, not token perplexity and not the 1.4440572213535412 validation BPB. The test result "
        "was obtained only after freezing the method and was not used to revise it."
    ))
    add_heading(doc, "Checkpoint identity", 2)
    make_table(doc,
        ["Artifact", "SHA256 or identifier"],
        [
            ["Final checkpoint", "9e7ae095e6a3da3589cf3e6857b058e08d7bac00a7022d9e82d50bb00e38a44c"],
            ["Student implementation", "68a9fdb1c678ec5ad0306b02115b8accb15fd0577f4dad9ad527d65cf77acec1"],
            ["Evaluator", "128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d"],
            ["Tokenizer", "020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e"],
            ["Protocol", "7506-mp1-wt2-v2"],
        ],
        widths=[4.8, 12.1],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
        font_size=7.3,
    )
    add_heading(doc, "Conclusion", 2)
    add_body(doc, (
        "The controlled 2 by 2 experiment supports both proposed mechanisms. Distillation and the causal "
        "cache each reduce validation BPB by about 0.063 when used alone, and the combined model reaches "
        "1.44406 at equal training targets. The mechanisms have partially overlapping effects, but each "
        "still contributes about 0.049 BPB in the presence of the other. The frozen checkpoint obtains "
        "1.4440572213535412 validation BPB and 1.4615442689258094 full-test BPB. The latter is the submitted "
        "course score. The final predictor satisfies the measured CPU time, memory, and asset limits, with "
        "CPU timing as the main remaining margin risk."
    ))
    add_heading(doc, "AI assistance disclosure", 2)
    add_body(doc, (
        "Substantive AI assistance was used to discuss architecture and experiment choices, implement optional "
        "training mechanisms, and draft and format this report. I verified the configurations, checkpoint "
        "metadata, numerical results, resource measurements, and compliance claims against the repository "
        "code and experiment logs."
    ))
    add_heading(doc, "References", 2)
    refs = [
        "1  DASE7506 MP1 Small Language Model Challenge guide and code README  2026",
        "2  Merity S  Xiong C  Bradbury J  and Socher R  Pointer Sentinel Mixture Models  2016",
        "3  Vaswani A et al  Attention Is All You Need  2017",
        "4  Su J et al  RoFormer Enhanced Transformer with Rotary Position Embedding  2021",
        "5  Zhang B and Sennrich R  Root Mean Square Layer Normalization  2019",
        "6  Shazeer N  GLU Variants Improve Transformer  2020",
        "7  Hinton G  Vinyals O  and Dean J  Distilling the Knowledge in a Neural Network  2015",
        "8  Grave E  Joulin A  and Usunier N  Improving Neural Language Models with a Continuous Cache  2017",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.25)
        p.paragraph_format.first_line_indent = Cm(-0.25)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(ref)
        set_run_font(r, size=7.2, color=CHARCOAL)

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
