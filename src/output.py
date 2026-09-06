import base64
import datetime as dt
import html
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import Assignment

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    PlaywrightError = RuntimeError
    sync_playwright = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


OUTPUT_DIR = Path("generated_submissions")
HANDWRITING_FONT_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "fonts"
    / "gveret_levin"
    / "GveretLevin-Regular.woff2"
)
HANDWRITING_VARIANT_PATHS = tuple(
    HANDWRITING_FONT_PATH.with_name(f"GveretLevin-Synthetic-{index}.woff2")
    for index in range(1, 5)
)
PERSONAL_HANDWRITING_VARIANT_PATHS = tuple(
    Path(__file__).resolve().parent.parent
    / "HandWrite"
    / "fonts"
    / f"RazPersonalHandwriting-{index}.woff2"
    for index in range(1, 5)
)
OUTPUT_STYLES = {"standard", "ipad_clean"}
PAPER_STYLES = {"plain", "grid", "ruled"}
HANDWRITING_FONTS = {"personal", "gveret"}
WINDOWS_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\calibri.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
]
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


@dataclass(frozen=True)
class RenderProfile:
    output_style: str = "standard"
    paper_style: str = "plain"
    synthetic_glyph_variation: bool = True
    handwriting_font: str = "personal"

    def __post_init__(self) -> None:
        if self.output_style not in OUTPUT_STYLES:
            raise ValueError(f"Unsupported output style: {self.output_style}")
        if self.paper_style not in PAPER_STYLES:
            raise ValueError(f"Unsupported paper style: {self.paper_style}")
        if self.handwriting_font not in HANDWRITING_FONTS:
            raise ValueError(f"Unsupported handwriting font: {self.handwriting_font}")


@lru_cache(maxsize=2)
def embedded_handwriting_font_css(handwriting_font: str = "personal") -> str:
    if handwriting_font == "personal":
        base_path = PERSONAL_HANDWRITING_VARIANT_PATHS[0]
        variant_paths = PERSONAL_HANDWRITING_VARIANT_PATHS
    elif handwriting_font == "gveret":
        base_path = HANDWRITING_FONT_PATH
        variant_paths = HANDWRITING_VARIANT_PATHS
    else:
        raise ValueError(f"Unsupported handwriting font: {handwriting_font}")
    required_paths = (base_path, *variant_paths)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Handwriting fonts are missing: {', '.join(missing)}")

    def face(font_path: Path, family: str) -> str:
        encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
        return (
            "    @font-face {\n"
            f"      font-family: '{family}';\n"
            f"      src: url(data:font/woff2;base64,{encoded}) format('woff2');\n"
            "      font-style: normal;\n"
            "      font-weight: 400;\n"
            "      font-display: block;\n"
            "    }\n"
        )

    return face(base_path, "Handwriting Base") + "".join(
        face(path, f"Handwriting Variant {index}")
        for index, path in enumerate(variant_paths, start=1)
    )


def ipad_clean_css(paper_style: str, handwriting_font: str = "personal") -> str:
    backgrounds = {
        "plain": "#fffefb",
        "grid": (
            "linear-gradient(rgba(92, 145, 190, 0.13) 1px, transparent 1px), "
            "linear-gradient(90deg, rgba(92, 145, 190, 0.13) 1px, transparent 1px), #fffefb"
        ),
        "ruled": "linear-gradient(rgba(92, 145, 190, 0.14) 1px, transparent 1px), #fffefb",
    }
    background_sizes = {
        "plain": "auto",
        "grid": "7mm 7mm, 7mm 7mm, auto",
        "ruled": "auto 8mm, auto",
    }
    return (
        embedded_handwriting_font_css(handwriting_font)
        + "    body.output-ipad-clean {\n"
        + "      color: #111111;\n"
        + f"      background: {backgrounds[paper_style]};\n"
        + f"      background-size: {background_sizes[paper_style]};\n"
        + "    }\n"
        + "    body.output-ipad-clean .page {\n"
        + f"      background: {backgrounds[paper_style]};\n"
        + f"      background-size: {background_sizes[paper_style]};\n"
        + "      color: #111111;\n"
        + "      font-family: 'Handwriting Base', 'Segoe UI', Arial, sans-serif;\n"
        + "      font-size: 1.16rem;\n"
        + "      line-height: 1.85;\n"
        + "    }\n"
        + "    body.output-ipad-clean h1, body.output-ipad-clean h2, "
        + "body.output-ipad-clean h3, body.output-ipad-clean h4 {\n"
        + "      color: #080808; font-family: 'Handwriting Base', 'Segoe UI', Arial, sans-serif;\n"
        + "      font-weight: 400; letter-spacing: 0.01em; break-after: avoid-page; page-break-after: avoid;\n"
        + "    }\n"
        + "    body.output-ipad-clean h3 { padding-inline: 2mm; }\n"
        + "    body.output-ipad-clean h2 { border-bottom: 2px solid rgba(17, 17, 17, 0.38); }\n"
        + "    body.output-ipad-clean p, body.output-ipad-clean li { font-size: 1.08em; }\n"
        + "    body.output-ipad-clean strong { color: #050505; font-weight: 700; }\n"
        + "    body.output-ipad-clean code, body.output-ipad-clean pre, "
        + "body.output-ipad-clean .ltr, body.output-ipad-clean .compound-card figcaption {\n"
        + "      font-family: 'Consolas', 'Courier New', monospace;\n"
        + "    }\n"
        + "    body.output-ipad-clean code, body.output-ipad-clean pre {\n"
        + "      color: #111111; background: rgba(235, 241, 246, 0.88);\n"
        + "    }\n"
        + "    body.output-ipad-clean .glyph-variant-1 { font-family: 'Handwriting Variant 1', 'Handwriting Base', sans-serif; }\n"
        + "    body.output-ipad-clean .glyph-variant-2 { font-family: 'Handwriting Variant 2', 'Handwriting Base', sans-serif; }\n"
        + "    body.output-ipad-clean .glyph-variant-3 { font-family: 'Handwriting Variant 3', 'Handwriting Base', sans-serif; }\n"
        + "    body.output-ipad-clean .glyph-variant-4 { font-family: 'Handwriting Variant 4', 'Handwriting Base', sans-serif; }\n"
        + "    body.output-ipad-clean[data-synthetic-glyph-variation='true'] .page { font-kerning: none; }\n"
        + "    body.output-ipad-clean mjx-container { color: #111111; }\n"
        + "    body.output-ipad-clean mjx-container[display='true'] {\n"
        + "      margin: 0.9rem 0; max-width: 100%; overflow: hidden;\n"
        + "    }\n"
        + "    body.output-ipad-clean mjx-container[display='true'] svg {\n"
        + "      max-width: 100% !important; min-width: 0 !important; height: auto !important;\n"
        + "    }\n"
        + "    body.output-ipad-clean .compound-references { color: #111111; }\n"
        + "    body.output-ipad-clean .compound-card {\n"
        + "      border: 0; border-radius: 0; background: transparent; box-shadow: none;\n"
        + "    }\n"
        + "    body.output-ipad-clean .molecule-drawing svg { background: transparent; }\n"
        + "    body.output-ipad-clean table { background: rgba(255, 254, 251, 0.78); }\n"
        + "    body.output-ipad-clean .preview-gallery, "
        + "body.output-ipad-clean .preview-gallery h2, "
        + "body.output-ipad-clean .preview-gallery figcaption {\n"
        + "      color: #1f1a17; font-family: 'Segoe UI', Arial, sans-serif; background: #fff;\n"
        + "    }\n"
    )


def sanitize_filename(value: str) -> str:
    normalized = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    normalized = re.sub(r"\s+", "_", normalized.strip())
    return normalized[:80] or "assignment"


def contains_non_latin_text(value: str) -> bool:
    return any(ord(char) > 255 for char in value)


def find_reportlab_font_path() -> Path | None:
    for candidate in WINDOWS_FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def build_answer_pdf_with_reportlab(pdf_path: Path, title: str, lines: list[str]) -> str:
    font_name = "Helvetica"
    font_path = find_reportlab_font_path()
    if font_path is not None:
        pdfmetrics.registerFont(TTFont("WorkflowFont", str(font_path)))
        font_name = "WorkflowFont"

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    body_style = styles["BodyText"].clone("WorkflowBody")
    body_style.fontName = font_name
    body_style.fontSize = 11
    body_style.leading = 15
    title_style = styles["Heading1"].clone("WorkflowTitle")
    title_style.fontName = font_name

    story = [Paragraph(html.escape(title), title_style), Spacer(1, 12)]
    for line in lines:
        if line.strip():
            story.append(Paragraph(html.escape(line), body_style))
        else:
            story.append(Spacer(1, 10))

    doc.build(story)
    return "reportlab"


def build_answer_pdf_basic(pdf_path: Path, title: str, lines: list[str]) -> str:
    page_width = 595
    page_height = 842
    left_margin = 50
    top_margin = 780
    line_height = 16
    usable_lines = 44

    content_streams: list[bytes] = []
    all_lines = [title, ""] + lines

    for start in range(0, len(all_lines), usable_lines):
        page_lines = all_lines[start : start + usable_lines]
        text_lines = ["BT", "/F1 12 Tf", f"1 0 0 1 {left_margin} {top_margin} Tm"]

        for index, raw_line in enumerate(page_lines):
            safe_line = raw_line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if index == 0:
                text_lines.append(f"({safe_line}) Tj")
            else:
                text_lines.append(f"0 -{line_height} Td")
                text_lines.append(f"({safe_line}) Tj")

        text_lines.append("ET")
        content_streams.append("\n".join(text_lines).encode("latin-1", errors="replace"))

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    page_ids = [4 + index * 2 for index in range(len(content_streams))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, stream in enumerate(content_streams):
        page_id = page_ids[index]
        content_id = page_id + 1
        page_object = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        )
        objects.append(page_object.encode())
        stream_object = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        objects.append(stream_object)

    pdf_bytes = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf_bytes))
        pdf_bytes.extend(f"{index} 0 obj\n".encode())
        pdf_bytes.extend(obj)
        pdf_bytes.extend(b"\nendobj\n")

    xref_offset = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(offsets)}\n".encode())
    pdf_bytes.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf_bytes.extend(f"{offset:010d} 00000 n \n".encode())

    trailer = f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    pdf_bytes.extend(trailer.encode())
    pdf_path.write_bytes(pdf_bytes)
    return "basic"


def build_answer_pdf(pdf_path: Path, title: str, lines: list[str]) -> str:
    all_text = "\n".join([title, *lines])
    if REPORTLAB_AVAILABLE:
        return build_answer_pdf_with_reportlab(pdf_path, title, lines)

    build_answer_pdf_basic(pdf_path, title, lines)
    if contains_non_latin_text(all_text):
        return "basic-nonlatin-risk"
    return "basic"


def wrap_answer_text(answer_text: str, width: int = 88) -> list[str]:
    lines: list[str] = []
    for block in answer_text.splitlines():
        block = block.rstrip()
        if not block:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(block, width=width) or [""])
    return lines


def infer_text_direction(value: str) -> str:
    return "rtl" if HEBREW_RE.search(value or "") else "ltr"


def strip_html_to_text(value: str) -> str:
    without_tags = re.sub(r"<br\s*/?>", "\n", value or "", flags=re.IGNORECASE)
    without_tags = re.sub(r"</(p|div|section|article|li|ul|ol|h[1-6])>", "\n", without_tags, flags=re.IGNORECASE)
    without_tags = re.sub(r"<[^>]+>", " ", without_tags)
    unescaped = html.unescape(without_tags)
    cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in unescaped.splitlines()]
    return "\n".join(line for line in cleaned_lines if line).strip()


def build_preview_gallery(preview_image_paths: list[str] | None) -> str:
    if not preview_image_paths:
        return ""

    figures: list[str] = []
    for index, raw_path in enumerate(preview_image_paths, start=1):
        image_name = html.escape(Path(raw_path).name)
        image_uri = Path(raw_path).resolve().as_uri()
        figures.append(
            "      <figure class=\"source-preview\">\n"
            f"        <img src=\"{image_uri}\" alt=\"Source preview page {index}\"/>\n"
            f"        <figcaption>Source PDF preview {index}: {image_name}</figcaption>\n"
            "      </figure>"
        )

    return (
        "    <section class=\"preview-gallery\" dir=\"ltr\">\n"
        "      <h2>Source Preview Images</h2>\n"
        + "\n".join(figures)
        + "\n    </section>\n"
    )


def build_rich_html_document(
    title: str,
    body_html: str,
    preview_image_paths: list[str] | None = None,
    *,
    output_style: str = "standard",
    paper_style: str = "plain",
    synthetic_glyph_variation: bool | None = None,
    handwriting_font: str = "personal",
) -> str:
    variation_requested = True if synthetic_glyph_variation is None else synthetic_glyph_variation
    profile = RenderProfile(output_style, paper_style, variation_requested, handwriting_font)
    escaped_title = html.escape(title)
    direction = infer_text_direction(f"{title}\n{body_html}")
    preview_gallery = build_preview_gallery(preview_image_paths)
    lang = "he" if direction == "rtl" else "en"
    body_class = "output-ipad-clean" if profile.output_style == "ipad_clean" else "output-standard"
    style_css = (
        ipad_clean_css(profile.paper_style, profile.handwriting_font)
        if profile.output_style == "ipad_clean"
        else ""
    )
    font_required = "true" if profile.output_style == "ipad_clean" else "false"
    variation_enabled = (
        "true"
        if profile.output_style == "ipad_clean" and profile.synthetic_glyph_variation
        else "false"
    )
    variation_seed = json.dumps(title, ensure_ascii=False)
    personal_latin_variation = "true" if profile.handwriting_font == "personal" else "false"

    return (
        "<!doctype html>\n"
        f"<html lang=\"{lang}\" dir=\"{direction}\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"  <title>{escaped_title}</title>\n"
        "  <style>\n"
        "    :root { color-scheme: light; }\n"
        "    * { box-sizing: border-box; }\n"
        "    body {\n"
        "      margin: 0;\n"
        "      padding: 0;\n"
        "      background: #f4f1ea;\n"
        "      color: #1f1a17;\n"
        "      font-family: 'Segoe UI', 'Arial', sans-serif;\n"
        "      line-height: 1.75;\n"
        "    }\n"
        "    main {\n"
        "      max-width: 860px;\n"
        "      margin: 0 auto;\n"
        "      padding: 32px 28px 48px;\n"
        "    }\n"
        "    .page {\n"
        "      background: #fffdf9;\n"
        "      border: 1px solid #ddd3c4;\n"
        "      border-radius: 18px;\n"
        "      box-shadow: 0 14px 40px rgba(49, 37, 18, 0.08);\n"
        "      padding: 34px 36px;\n"
        "    }\n"
        "    h1, h2, h3 { line-height: 1.3; margin: 0 0 0.8rem; }\n"
        "    h1 { font-size: 2rem; }\n"
        "    h2 { font-size: 1.35rem; margin-top: 1.8rem; }\n"
        "    p, li { font-size: 1rem; }\n"
        "    table { width: 100%; border-collapse: collapse; margin: 1rem 0; }\n"
        "    th, td { border: 1px solid #d8cfbf; padding: 0.55rem; text-align: start; vertical-align: top; }\n"
        "    th { background: #f5efe4; }\n"
        "    ul, ol { padding-inline-start: 1.4rem; }\n"
        "    .document-body { unicode-bidi: plaintext; }\n"
        "    .question-answer { min-width: 0; max-width: 100%; }\n"
        "    .question-answer p, .question-answer li {\n"
        "      max-width: 100%;\n"
        "      overflow-wrap: anywhere;\n"
        "      word-break: normal;\n"
        "      unicode-bidi: plaintext;\n"
        "    }\n"
        "    mjx-container, mjx-container svg { max-width: 100%; }\n"
        "    mjx-assistive-mml { display: none !important; }\n"
        "    mjx-container[display=\"true\"] { overflow-x: auto; overflow-y: hidden; }\n"
        "    .compound-references { direction: ltr; margin: 1rem 0; page-break-inside: avoid; }\n"
        "    .compound-card { display: inline-flex; width: min(100%, 390px); margin: 0.5rem; padding: 0.75rem; "
        "border: 1px solid #d8cfbf; border-radius: 12px; background: #fff; flex-direction: column; "
        "align-items: center; vertical-align: top; page-break-inside: avoid; }\n"
        "    .compound-card svg { display: block; max-width: 100%; height: auto; }\n"
        "    .compound-card figcaption { display: flex; gap: 0.65rem; align-items: center; flex-wrap: wrap; }\n"
        "    .compound-card-invalid { color: #8a2b20; align-items: flex-start; }\n"
        "    .compound-card-invalid span { font-family: 'Consolas', monospace; }\n"
        "    .ltr, code, pre, .math-inline, .math-block, .preview-gallery { direction: ltr; unicode-bidi: embed; }\n"
        "    code, pre {\n"
        "      font-family: 'Consolas', 'Courier New', monospace;\n"
        "      background: #f5efe4;\n"
        "      border-radius: 10px;\n"
        "    }\n"
        "    code { padding: 0.12rem 0.35rem; }\n"
        "    pre {\n"
        "      padding: 1rem;\n"
        "      overflow-x: auto;\n"
        "      white-space: pre-wrap;\n"
        "    }\n"
        "    img {\n"
        "      display: block;\n"
        "      max-width: 100%;\n"
        "      height: auto;\n"
        "      border-radius: 14px;\n"
        "      border: 1px solid #d8cfbf;\n"
        "      page-break-inside: avoid;\n"
        "    }\n"
        "    .preview-gallery { break-before: page; page-break-before: always; }\n"
        "    .source-preview {\n"
        "      margin: 0 0 12mm;\n"
        "      break-inside: avoid-page;\n"
        "      page-break-inside: avoid;\n"
        "    }\n"
        "    .source-preview + .source-preview { break-before: page; page-break-before: always; }\n"
        "    .source-preview img { width: auto; max-height: 220mm; margin: 0 auto; object-fit: contain; }\n"
        "    .source-preview figcaption {\n"
        "      margin-top: 0.4rem;\n"
        "      color: #675b4d;\n"
        "      font-size: 0.92rem;\n"
        "    }\n"
        "    .math-block {\n"
        "      margin: 1rem 0;\n"
        "      overflow-x: auto;\n"
        "    }\n"
        "    @page { size: A4; margin: 16mm 12mm 16mm; }\n"
        "    @page source-preview { size: A4; margin: 16mm 12mm 16mm; background: #fff; }\n"
        "    @media print {\n"
        "      body { background: #ffffff; }\n"
        "      main { max-width: none; padding: 0; }\n"
        "      .page { border: none; border-radius: 0; box-shadow: none; padding: 0; }\n"
        "      .question-answer, .question-answer p, .question-answer li { overflow: visible; }\n"
        "      .preview-gallery { page: source-preview; }\n"
        "    }\n"
        f"{style_css}"
        "  </style>\n"
        "  <script>\n"
        "    window.__mathJaxReady = false;\n"
        "    window.__fontReady = null;\n"
        "    window.__glyphVariantsReady = null;\n"
        f"    window.__handwritingFontRequired = {font_required};\n"
        f"    window.__syntheticGlyphVariation = {variation_enabled};\n"
        f"    window.__glyphVariationSeed = {variation_seed};\n"
        f"    window.__personalLatinGlyphs = {personal_latin_variation};\n"
        "    window.fitDisplayMath = () => {\n"
        "      document.querySelectorAll('mjx-container[display=\"true\"]').forEach((container) => {\n"
        "        const svg = container.querySelector('svg');\n"
        "        const available = container.parentElement ? container.parentElement.clientWidth : container.clientWidth;\n"
        "        if (svg && available > 0 && svg.getBoundingClientRect().width > available) {\n"
        "          svg.style.width = '100%';\n"
        "          svg.style.height = 'auto';\n"
        "          container.classList.add('math-scaled-to-page');\n"
        "        }\n"
        "      });\n"
        "    };\n"
        "    window.applySyntheticGlyphVariants = () => {\n"
        "      if (!window.__syntheticGlyphVariation) { window.__glyphVariantsReady = true; return; }\n"
        "      if (typeof Intl.Segmenter !== 'function') { throw new Error('Intl.Segmenter is unavailable'); }\n"
        "      const segmenter = new Intl.Segmenter('he', { granularity: 'grapheme' });\n"
        "      const personalGlyphPattern = window.__personalLatinGlyphs\n"
        "        ? /[A-Za-z\\u03A0\\u03A3\\u03B8\\u05D0-\\u05EA]/u\n"
        "        : /[\\u05D0-\\u05EA]/u;\n"
        "      const excluded = 'mjx-container, code, pre, .ltr, .math-inline, .math-block, '\n"
        "        + '.compound-references, .preview-gallery, svg, script, style, .glyph-variant';\n"
        "      const roots = [document.querySelector('.page > h1'), document.querySelector('.document-body')]\n"
        "        .filter(Boolean);\n"
        "      const textNodes = [];\n"
        "      roots.forEach((root) => {\n"
        "        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);\n"
        "        let node;\n"
        "        while ((node = walker.nextNode())) {\n"
        "          const parent = node.parentElement;\n"
        "          if (parent && !parent.closest(excluded) && personalGlyphPattern.test(node.nodeValue || '')) {\n"
        "            textNodes.push(node);\n"
        "          }\n"
        "        }\n"
        "      });\n"
        "      const hash32 = (value) => {\n"
        "        let hash = 0x811c9dc5;\n"
        "        for (let index = 0; index < value.length; index += 1) {\n"
        "          hash ^= value.charCodeAt(index);\n"
        "          hash = Math.imul(hash, 0x01000193);\n"
        "        }\n"
        "        return hash >>> 0;\n"
        "      };\n"
        "      let documentPosition = 0;\n"
        "      textNodes.forEach((textNode) => {\n"
        "        const owner = textNode.parentElement?.closest('[data-question-id], .question-answer');\n"
        "        const questionId = owner?.getAttribute('data-question-id') || owner?.id || 'document';\n"
        "        const fragment = document.createDocumentFragment();\n"
        "        for (const part of segmenter.segment(textNode.nodeValue || '')) {\n"
        "          const grapheme = part.segment;\n"
        "          if (personalGlyphPattern.test(grapheme)) {\n"
        "            const variant = (hash32(`${window.__glyphVariationSeed}|${questionId}|${documentPosition}|${grapheme}`) % 4) + 1;\n"
        "            const span = document.createElement('span');\n"
        "            span.className = `glyph-variant glyph-variant-${variant}`;\n"
        "            span.dataset.glyphVariant = String(variant);\n"
        "            span.textContent = grapheme;\n"
        "            fragment.appendChild(span);\n"
        "          } else {\n"
        "            fragment.appendChild(document.createTextNode(grapheme));\n"
        "          }\n"
        "          documentPosition += 1;\n"
        "        }\n"
        "        textNode.replaceWith(fragment);\n"
        "      });\n"
        "      window.__glyphVariantsReady = true;\n"
        "    };\n"
        "    window.addEventListener('DOMContentLoaded', () => {\n"
        "      const families = window.__syntheticGlyphVariation\n"
        "        ? ['Handwriting Base', 'Handwriting Variant 1', 'Handwriting Variant 2', "
        "'Handwriting Variant 3', 'Handwriting Variant 4']\n"
        "        : ['Handwriting Base'];\n"
        "      const loads = window.__handwritingFontRequired\n"
        "        ? families.map((family) => document.fonts.load(`16px \"${family}\"`, 'אבגדה ABC xyz ΣΠθ'))\n"
        "        : [];\n"
        "      Promise.all(loads).then(() => document.fonts.ready).then(() => {\n"
        "        window.__fontReady = !window.__handwritingFontRequired "
        "|| families.every((family) => document.fonts.check(`16px \"${family}\"`));\n"
        "      }).catch(() => { window.__fontReady = false; });\n"
        "    });\n"
        "    window.MathJax = {\n"
        "      tex: { inlineMath: [['\\\\(', '\\\\)'], ['$', '$']], displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']] },\n"
        "      svg: { fontCache: 'global' },\n"
        "      options: { enableAssistiveMml: false },\n"
        "      startup: {\n"
        "        ready: () => {\n"
        "          MathJax.startup.defaultReady();\n"
        "          MathJax.startup.promise.then(() => {\n"
        "            window.fitDisplayMath();\n"
        "            document.querySelectorAll('mjx-assistive-mml').forEach((element) => element.remove());\n"
        "            try {\n"
        "              window.applySyntheticGlyphVariants();\n"
        "              window.__mathJaxReady = true;\n"
        "            } catch (error) {\n"
        "              window.__glyphVariantsReady = false;\n"
        "              window.__glyphVariantError = String(error);\n"
        "            }\n"
        "          });\n"
        "        }\n"
        "      }\n"
        "    };\n"
        "  </script>\n"
        "  <script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js\"></script>\n"
        "</head>\n"
        f"<body class=\"{body_class} paper-{profile.paper_style}\" "
        f"data-output-style=\"{profile.output_style}\" data-paper-style=\"{profile.paper_style}\" "
        f"data-handwriting-font=\"{profile.handwriting_font}\" "
        f"data-synthetic-glyph-variation=\"{variation_enabled}\">\n"
        "  <main>\n"
        "    <article class=\"page\">\n"
        f"      <h1>{escaped_title}</h1>\n"
        f"      <div class=\"document-body\" dir=\"{direction}\">\n"
        f"{body_html.strip()}\n"
        "      </div>\n"
        f"{preview_gallery}"
        "    </article>\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


def write_text_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_html_file(
    path: Path,
    title: str,
    body_html: str,
    preview_image_paths: list[str] | None = None,
    *,
    output_style: str = "standard",
    paper_style: str = "plain",
    synthetic_glyph_variation: bool | None = None,
    handwriting_font: str = "personal",
) -> str:
    document = build_rich_html_document(
        title,
        body_html,
        preview_image_paths=preview_image_paths,
        output_style=output_style,
        paper_style=paper_style,
        synthetic_glyph_variation=synthetic_glyph_variation,
        handwriting_font=handwriting_font,
    )
    path.write_text(document, encoding="utf-8")
    return document


def render_html_to_pdf(
    html_path: Path,
    pdf_path: Path,
    title: str,
    fallback_text: str,
    *,
    output_style: str = "standard",
) -> str:
    if PLAYWRIGHT_AVAILABLE and sync_playwright is not None:
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page()
                page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
                page.wait_for_function("() => window.__fontReady !== null", timeout=5000)
                font_ready = page.evaluate("() => window.__fontReady")
                if output_style == "ipad_clean" and font_ready is not True:
                    raise RuntimeError("one or more selected handwriting fonts failed to load")
                try:
                    page.wait_for_function(
                        "() => window.__mathJaxReady === true "
                        "&& window.__glyphVariantsReady === true",
                        timeout=15000,
                    )
                except PlaywrightError as exc:
                    if output_style == "ipad_clean":
                        glyph_error = page.evaluate("() => window.__glyphVariantError || null")
                        raise RuntimeError(
                            f"MathJax or synthetic glyph variation did not finish: {glyph_error or exc}"
                        ) from exc
                    page.wait_for_timeout(1200)
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    margin={"top": "14mm", "right": "12mm", "bottom": "14mm", "left": "12mm"},
                )
                browser.close()
            return "playwright"
        except Exception as exc:
            if output_style == "ipad_clean":
                raise RuntimeError(f"ipad_clean PDF rendering failed: {exc}") from exc

    if output_style == "ipad_clean":
        raise RuntimeError("ipad_clean PDF rendering requires Playwright and Chromium")

    fallback_engine = build_answer_pdf(pdf_path, title, wrap_answer_text(fallback_text))
    return f"{fallback_engine}-fallback"


def print_json_output(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8"))


def write_routing_report(
    output_dir: Path,
    summary: dict[str, Any],
    *,
    basename: str = "routing_report",
) -> tuple[Path, Path]:
    report_results = []
    for result in summary.get("results", []):
        report_results.append(
            {
                "course": result.get("course"),
                "assignment": result.get("assignment"),
                "assignment_id": result.get("assignment_id"),
                "status": result.get("status"),
                "classified_topic": result.get("classified_topic"),
                "classification_confidence": result.get("classification_confidence"),
                "classification_method": result.get("classification_method"),
                "classification_reason": result.get("classification_reason"),
                "subject_agent": result.get("subject_agent"),
                "llm_provider": result.get("llm_provider"),
                "llm_model": result.get("llm_model"),
                "llm_temperature": result.get("llm_temperature"),
                "llm_reasoning_effort": result.get("llm_reasoning_effort"),
                "coverage_status": result.get("coverage_status"),
                "coverage_expected_count": result.get("coverage_expected_count"),
                "coverage_answered_count": result.get("coverage_answered_count"),
                "coverage_missing_ids": result.get("coverage_missing_ids", []),
                "submitted_now": result.get("submitted_now", False),
                "pdf_path": result.get("pdf_path"),
                "manifest_path": result.get("manifest_path"),
                "manifest_validation_path": result.get("manifest_validation_path"),
                "manifest_raw_path": result.get("manifest_raw_path"),
                "manifest_retry_raw_path": result.get("manifest_retry_raw_path"),
                "manifest_from_cache": result.get("manifest_from_cache"),
                "vision_request_count": result.get("vision_request_count", 0),
                "fallback_crop_count": result.get("fallback_crop_count", 0),
                "manifest_retry_count": result.get("manifest_retry_count", 0),
                "retried_pages": result.get("retried_pages", []),
                "resumed_from_checkpoint": result.get("resumed_from_checkpoint", False),
                "checkpoint_path": result.get("checkpoint_path"),
                "output_style": result.get("output_style", "standard"),
                "paper_style": result.get("paper_style", "plain"),
                "handwriting_font": result.get("handwriting_font", "personal"),
                "synthetic_glyph_variation": result.get(
                    "synthetic_glyph_variation", False
                ),
                "llm_usage": result.get("llm_usage", {}),
                "failed_stage": result.get("failed_stage"),
                "openai_status_code": result.get("openai_status_code"),
                "openai_error_type": result.get("openai_error_type"),
                "openai_error_code": result.get("openai_error_code"),
                "openai_request_id": result.get("openai_request_id"),
                "error": result.get("error"),
            }
        )

    report_payload = {
        "generated_at": summary.get("generated_at"),
        "reference_date": summary.get("reference_date"),
        "topic_counts": summary.get("topic_counts", {}),
        "assignment_count": summary.get("assignment_count", len(report_results)),
        "error_count": summary.get("error_count", 0),
        "incomplete_count": summary.get("incomplete_count", 0),
        "llm_usage": summary.get("llm_usage", {}),
        "results": report_results,
        "skipped": summary.get("skipped", []),
    }
    safe_basename = sanitize_filename(basename)
    json_path = output_dir / f"{safe_basename}.json"
    json_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def cell(value: Any) -> str:
        if value is None or value == "":
            return "—"
        if isinstance(value, bool):
            return "כן" if value else "לא"
        return html.escape(str(value))

    rows: list[str] = []
    for result in report_results:
        confidence = result.get("classification_confidence")
        if isinstance(confidence, (int, float)):
            confidence = f"{confidence:.0%}"
        model = result.get("llm_model")
        provider = result.get("llm_provider")
        llm_label = f"{provider} / {model}" if provider and model else provider or model
        coverage_label = (
            f"{result.get('coverage_status') or '—'} "
            f"({result.get('coverage_answered_count') or 0}/{result.get('coverage_expected_count') or 0})"
        )
        missing_ids = result.get("coverage_missing_ids") or []
        if missing_ids:
            coverage_label += f"; missing: {', '.join(str(item) for item in missing_ids)}"
        failure_label = result.get("failed_stage")
        if result.get("openai_status_code"):
            failure_label = f"{failure_label or 'unknown'} / HTTP {result['openai_status_code']}"
        if result.get("openai_request_id"):
            failure_label = f"{failure_label or 'unknown'} / {result['openai_request_id']}"
        usage = result.get("llm_usage") or {}
        usage_label = (
            f"{usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out; "
            f"{usage.get('reasoning_tokens', 0)} reasoning / {usage.get('cached_tokens', 0)} cached; "
            f"${float(usage.get('estimated_cost_usd', 0)):.4f}; "
            f"vision={result.get('vision_request_count', 0)}, "
            f"manifest retries={result.get('manifest_retry_count', 0)}, "
            f"crops={result.get('fallback_crop_count', 0)}"
        )
        render_label = (
            f"{result.get('output_style', 'standard')} / "
            f"{result.get('paper_style', 'plain')} / "
            f"font={result.get('handwriting_font', 'personal')}"
        )
        if result.get("synthetic_glyph_variation"):
            variation_label = (
                "personal glyph variation"
                if result.get("handwriting_font", "personal") == "personal"
                else "synthetic glyph variation"
            )
            render_label += f" / {variation_label}"
        rows.append(
            "<tr>"
            f"<td>{cell(result.get('course'))}</td>"
            f"<td>{cell(result.get('assignment'))}</td>"
            f"<td>{cell(result.get('status'))}</td>"
            f"<td><code>{cell(result.get('classified_topic'))}</code></td>"
            f"<td>{cell(confidence)}</td>"
            f"<td>{cell(result.get('classification_method'))}</td>"
            f"<td>{cell(result.get('classification_reason') or result.get('error'))}</td>"
            f"<td dir=\"ltr\">{cell(failure_label)}</td>"
            f"<td>{cell(result.get('subject_agent'))}</td>"
            f"<td dir=\"ltr\">{cell(llm_label)}</td>"
            f"<td dir=\"ltr\">{cell(render_label)}</td>"
            f"<td dir=\"ltr\">{cell(usage_label)}</td>"
            f"<td>{cell(coverage_label)}</td>"
            f"<td>{cell(result.get('submitted_now'))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="14">לא עובדו מטלות בריצה זו.</td></tr>')

    skipped_rows: list[str] = []
    for skipped in summary.get("skipped", []):
        skipped_rows.append(
            "<tr>"
            f"<td>{cell(skipped.get('course'))}</td>"
            f"<td>{cell(skipped.get('assignment'))}</td>"
            f"<td>{cell(skipped.get('due_at'))}</td>"
            f"<td>{cell(skipped.get('reason'))}</td>"
            "</tr>"
        )

    body_html = (
        "<section>"
        f"<p>תאריך הרצה: <span dir=\"ltr\">{cell(summary.get('generated_at'))}</span></p>"
        f"<p>מספר מטלות שעובדו: {cell(summary.get('assignment_count', len(report_results)))} | "
        f"שגיאות: {cell(summary.get('error_count', 0))} | "
        f"חלקיות: {cell(summary.get('incomplete_count', 0))}</p>"
        f"<p dir=\"ltr\">LLM usage: {cell(summary.get('llm_usage', {}))}</p>"
        "<h2>סיווג וניתוב לסוכנים</h2>"
        "<table><thead><tr>"
        "<th>קורס</th><th>מטלה</th><th>סטטוס</th><th>סיווג</th><th>ביטחון</th>"
        "<th>שיטה</th><th>סיבת הסיווג / שגיאה</th><th>שלב כשל</th>"
        "<th>סוכן מומחה</th><th>LLM</th><th>עיצוב</th><th>Usage</th><th>כיסוי</th><th>הוגש</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    if skipped_rows:
        body_html += (
            "<h2>מטלות שדולגו</h2><table><thead><tr>"
            "<th>קורס</th><th>מטלה</th><th>מועד</th><th>סיבה</th>"
            "</tr></thead><tbody>"
            + "".join(skipped_rows)
            + "</tbody></table>"
        )
    body_html += "</section>"

    html_path = output_dir / f"{safe_basename}.html"
    write_html_file(html_path, "דוח סיווג וניתוב מטלות", body_html)
    return json_path, html_path


def ensure_output_dir(_: dt.date) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def build_assignment_output_dir(
    reference_date: dt.date,
    assignment: Assignment,
    *,
    run_label: str | None = None,
) -> Path:
    folder_name = (
        f"{reference_date.isoformat()}_"
        f"{sanitize_filename(assignment.course_name)}_"
        f"{sanitize_filename(assignment.title)}"
    )
    if run_label and run_label.strip():
        folder_name += f"_{sanitize_filename(run_label)}"
    assignment_dir = OUTPUT_DIR / folder_name
    assignment_dir.mkdir(parents=True, exist_ok=True)
    return assignment_dir
