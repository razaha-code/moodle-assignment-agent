"""Extract vector glyph candidates from the personal handwriting PDF.

The source PDF stores the ink as filled black outlines. This script keeps those
outlines as SVG paths and uses the notebook grid only to locate cells; it never
traces a raster preview or changes the source PDF.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz


CELL_WIDTH = 18.0
CELL_HEIGHT = 36.0
X_CENTERS_RTL = tuple(477.0 - 18.0 * index for index in range(11))
PAGE_1_Y_CENTERS = tuple(66.0 + 36.0 * index for index in range(22))
PAGE_2_Y_CENTERS = tuple(30.0 + 36.0 * index for index in range(19))
HEBREW_LETTERS = tuple("אבגדהוזחטיכלמנסעפצקרשת")
PAGE_2_LABELS: tuple[str | None, ...] = (
    "ץ",
    "ף",
    "ם",
    ".",
    ",",
    ")",
    "(",
    "?",
    "!",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)
SUPPLEMENTAL_FINAL_FILENAME = "final_kaf_nun_samples_2026-08-21.pdf"
SUPPLEMENTAL_X_CENTERS_RTL = tuple(466.0 - 28.0 * index for index in range(10))
SUPPLEMENTAL_ROWS = (
    {"label": "ן", "top": 60.0},
    {"label": "ך", "top": 114.0},
)
SUPPLEMENTAL_CELL_WIDTH = 28.0
SUPPLEMENTAL_CELL_HEIGHT = 54.0
ENGLISH_SYMBOLS_FILENAME = "english_symbols_samples_2026-08-24.pdf"
ENGLISH_SOURCE_ID = "english_symbols_2026_08_24"
ENGLISH_X_CENTERS = (72.0, 100.0, 128.0, 156.0)
# The page contains A-N, one additional lowercase n row, and O-Z.
ENGLISH_PAGE_1_LABELS = tuple("ABCDEFGHIJKLMN") + ("n",) + tuple("OPQRSTUVWXYZ")
ENGLISH_PAGE_1_Y_CENTERS = (
    76.5, 104.0, 132.0, 158.0, 186.0, 215.0, 243.0, 270.0, 299.0,
    327.0, 354.0, 380.0, 414.0, 442.0, 471.0, 497.0, 525.0, 554.0,
    585.0, 613.0, 640.0, 672.0, 699.0, 726.0, 752.0, 784.0, 811.0,
)
SYMBOL_X_CENTERS = (68.0, 100.0, 130.0, 158.0, 188.0)
SYMBOL_PAGE_2_LABELS = ("Σ", "Π", "θ")
SYMBOL_PAGE_2_Y_CENTERS = (18.0, 73.0, 137.0)


@dataclass(frozen=True)
class VectorSubpath:
    items: tuple[tuple[Any, ...], ...]
    bbox: fitz.Rect

    @property
    def center(self) -> fitz.Point:
        return fitz.Point((self.bbox.x0 + self.bbox.x1) / 2, (self.bbox.y0 + self.bbox.y1) / 2)


def _point_close(left: fitz.Point, right: fitz.Point, tolerance: float = 0.05) -> bool:
    return abs(left.x - right.x) <= tolerance and abs(left.y - right.y) <= tolerance


def _item_start(item: tuple[Any, ...]) -> fitz.Point:
    if item[0] in {"l", "c"}:
        return item[1]
    if item[0] == "re":
        return item[1].tl
    if item[0] == "qu":
        return item[1].ul
    raise ValueError(f"Unsupported PDF path item: {item[0]}")


def _item_end(item: tuple[Any, ...]) -> fitz.Point:
    if item[0] == "l":
        return item[2]
    if item[0] == "c":
        return item[4]
    if item[0] == "re":
        return item[1].tl
    if item[0] == "qu":
        return item[1].ul
    raise ValueError(f"Unsupported PDF path item: {item[0]}")


def _item_points(item: tuple[Any, ...]) -> Iterable[fitz.Point]:
    if item[0] in {"l", "c"}:
        yield from item[1:]
    elif item[0] == "re":
        rect = item[1]
        yield from (rect.tl, rect.tr, rect.br, rect.bl)
    elif item[0] == "qu":
        quad = item[1]
        yield from (quad.ul, quad.ur, quad.lr, quad.ll)
    else:
        raise ValueError(f"Unsupported PDF path item: {item[0]}")


def _subpath_bbox(items: list[tuple[Any, ...]]) -> fitz.Rect:
    points = [point for item in items for point in _item_points(item)]
    return fitz.Rect(
        min(point.x for point in points),
        min(point.y for point in points),
        max(point.x for point in points),
        max(point.y for point in points),
    )


def split_compound_path(items: list[tuple[Any, ...]]) -> list[VectorSubpath]:
    subpaths: list[VectorSubpath] = []
    current: list[tuple[Any, ...]] = []
    previous_end: fitz.Point | None = None
    for item in items:
        start = _item_start(item)
        if current and previous_end is not None and not _point_close(start, previous_end):
            subpaths.append(VectorSubpath(tuple(current), _subpath_bbox(current)))
            current = []
        current.append(item)
        previous_end = _item_end(item)
    if current:
        subpaths.append(VectorSubpath(tuple(current), _subpath_bbox(current)))
    return subpaths


def black_ink_subpaths(page: fitz.Page) -> tuple[list[VectorSubpath], bool]:
    black_drawings = [
        drawing
        for drawing in page.get_drawings()
        if drawing.get("fill") is not None and max(drawing["fill"]) <= 0.05
    ]
    if len(black_drawings) != 1:
        raise ValueError(f"Expected one black compound ink path, found {len(black_drawings)}")
    drawing = black_drawings[0]
    return split_compound_path(drawing["items"]), bool(drawing.get("even_odd"))


def _fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def subpath_to_svg_path(
    subpath: VectorSubpath,
    offset_x: float,
    offset_y: float,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> str:
    commands: list[str] = []
    current: fitz.Point | None = None
    for item in subpath.items:
        kind = item[0]
        start = _item_start(item)
        if current is None or not _point_close(start, current):
            commands.append(
                f"M {_fmt((start.x - offset_x) * scale_x)} {_fmt((start.y - offset_y) * scale_y)}"
            )
        if kind == "l":
            end = item[2]
            commands.append(
                f"L {_fmt((end.x - offset_x) * scale_x)} {_fmt((end.y - offset_y) * scale_y)}"
            )
        elif kind == "c":
            control_1, control_2, end = item[2], item[3], item[4]
            commands.append(
                "C "
                f"{_fmt((control_1.x - offset_x) * scale_x)} {_fmt((control_1.y - offset_y) * scale_y)} "
                f"{_fmt((control_2.x - offset_x) * scale_x)} {_fmt((control_2.y - offset_y) * scale_y)} "
                f"{_fmt((end.x - offset_x) * scale_x)} {_fmt((end.y - offset_y) * scale_y)}"
            )
        elif kind == "re":
            rect = item[1]
            commands.extend(
                (
                    f"L {_fmt((rect.x1 - offset_x) * scale_x)} {_fmt((rect.y0 - offset_y) * scale_y)}",
                    f"L {_fmt((rect.x1 - offset_x) * scale_x)} {_fmt((rect.y1 - offset_y) * scale_y)}",
                    f"L {_fmt((rect.x0 - offset_x) * scale_x)} {_fmt((rect.y1 - offset_y) * scale_y)}",
                    "Z",
                )
            )
        elif kind == "qu":
            quad = item[1]
            commands.extend(
                (
                    f"L {_fmt((quad.ur.x - offset_x) * scale_x)} {_fmt((quad.ur.y - offset_y) * scale_y)}",
                    f"L {_fmt((quad.lr.x - offset_x) * scale_x)} {_fmt((quad.lr.y - offset_y) * scale_y)}",
                    f"L {_fmt((quad.ll.x - offset_x) * scale_x)} {_fmt((quad.ll.y - offset_y) * scale_y)}",
                    "Z",
                )
            )
        current = _item_end(item)
    if commands and commands[-1] != "Z":
        commands.append("Z")
    return " ".join(commands)


def glyph_slug(label: str | None, page_number: int, row_number: int) -> str:
    if label is None:
        return f"page_{page_number}_unmapped_row_{row_number:02d}"
    if len(label) == 1 and "א" <= label <= "ת":
        return f"u{ord(label):04x}"
    names = {".": "period", ",": "comma", ")": "paren_close", "(": "paren_open", "?": "question", "!": "exclamation"}
    return names.get(label, f"digit_{label}")


def unicode_glyph_slug(label: str) -> str:
    if len(label) != 1:
        raise ValueError(f"Expected one Unicode character, got {label!r}")
    return f"u{ord(label):04x}"


def _svg_document(paths: list[str], even_odd: bool) -> str:
    rule = "evenodd" if even_odd else "nonzero"
    path_markup = " ".join(f'<path d="{html.escape(path, quote=True)}"/>' for path in paths)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 18 36" '
        'width="18" height="36" role="img">'
        f'<g fill="#000" fill-rule="{rule}">{path_markup}</g></svg>\n'
    )


def extract_page(
    page: fitz.Page,
    page_number: int,
    y_centers: tuple[float, ...],
    labels: tuple[str | None, ...],
    output_dir: Path,
) -> list[dict[str, Any]]:
    subpaths, even_odd = black_ink_subpaths(page)
    rows: list[dict[str, Any]] = []
    for row_index, (center_y, label) in enumerate(zip(y_centers, labels), start=1):
        slug = glyph_slug(label, page_number, row_index)
        row_dir = output_dir / slug
        row_dir.mkdir(parents=True, exist_ok=True)
        samples: list[dict[str, Any]] = []
        for sample_index, center_x in enumerate(X_CENTERS_RTL, start=1):
            left = center_x - CELL_WIDTH / 2
            top = center_y - CELL_HEIGHT / 2
            cell_paths = [
                subpath
                for subpath in subpaths
                if left <= subpath.center.x < left + CELL_WIDTH
                and top <= subpath.center.y < top + CELL_HEIGHT
            ]
            if not cell_paths:
                continue
            svg_paths = [subpath_to_svg_path(path, left, top) for path in cell_paths]
            svg_path = row_dir / f"sample_{sample_index:02d}.svg"
            svg_path.write_text(_svg_document(svg_paths, even_odd), encoding="utf-8")
            bbox = fitz.Rect()
            for path in cell_paths:
                bbox.include_rect(path.bbox)
            samples.append(
                {
                    "sample_index": sample_index,
                    "file": svg_path.relative_to(output_dir.parent).as_posix(),
                    "source_cell": {"center_x": center_x, "center_y": center_y},
                    "subpath_count": len(cell_paths),
                    "bbox": [round(value, 3) for value in (bbox.x0, bbox.y0, bbox.x1, bbox.y1)],
                }
            )
        rows.append(
            {
                "page": page_number,
                "row": row_index,
                "label": label,
                "slug": slug,
                "mapping_status": (
                    "user_confirmed"
                    if page_number == 2 and row_index <= 3
                    else "confirmed_layout"
                    if label is not None
                    else "needs_user_review"
                ),
                "sample_count": len(samples),
                "samples": samples,
            }
        )
    return rows


def extract_supplemental_final_letters(
    source_pdf: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    with fitz.open(source_pdf) as document:
        if document.page_count != 1:
            raise ValueError(f"Expected one supplemental page, found {document.page_count}")
        subpaths, even_odd = black_ink_subpaths(document[0])

    scale_x = CELL_WIDTH / SUPPLEMENTAL_CELL_WIDTH
    scale_y = CELL_HEIGHT / SUPPLEMENTAL_CELL_HEIGHT
    rows: list[dict[str, Any]] = []
    for row_index, row_spec in enumerate(SUPPLEMENTAL_ROWS, start=1):
        label = row_spec["label"]
        top = row_spec["top"]
        slug = glyph_slug(label, 3, row_index)
        row_dir = output_dir / slug
        row_dir.mkdir(parents=True, exist_ok=True)
        samples: list[dict[str, Any]] = []
        for sample_index, center_x in enumerate(SUPPLEMENTAL_X_CENTERS_RTL, start=1):
            left = center_x - SUPPLEMENTAL_CELL_WIDTH / 2
            cell_paths = [
                subpath
                for subpath in subpaths
                if left <= subpath.center.x < left + SUPPLEMENTAL_CELL_WIDTH
                and top <= subpath.center.y < top + SUPPLEMENTAL_CELL_HEIGHT
            ]
            if not cell_paths:
                continue
            svg_paths = [
                subpath_to_svg_path(
                    path,
                    left,
                    top,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                for path in cell_paths
            ]
            svg_path = row_dir / f"sample_{sample_index:02d}.svg"
            svg_path.write_text(_svg_document(svg_paths, even_odd), encoding="utf-8")
            bbox = fitz.Rect()
            for path in cell_paths:
                bbox.include_rect(path.bbox)
            canonical_bbox = [
                (bbox.x0 - left) * scale_x,
                (bbox.y0 - top) * scale_y,
                (bbox.x1 - left) * scale_x,
                (bbox.y1 - top) * scale_y,
            ]
            samples.append(
                {
                    "sample_index": sample_index,
                    "file": svg_path.relative_to(output_dir.parent).as_posix(),
                    "source_file": f"input/{SUPPLEMENTAL_FINAL_FILENAME}",
                    "source_cell": {"center_x": center_x, "top": top},
                    "subpath_count": len(cell_paths),
                    "bbox": [round(value, 3) for value in canonical_bbox],
                }
            )
        rows.append(
            {
                "page": 3,
                "row": row_index,
                "label": label,
                "slug": slug,
                "mapping_status": "user_confirmed_supplemental",
                "sample_count": len(samples),
                "samples": samples,
            }
        )
    return rows


def extract_supplemental_grid_page(
    page: fitz.Page,
    *,
    manifest_page: int,
    labels: tuple[str, ...],
    x_centers: tuple[float, ...],
    y_centers: tuple[float, ...],
    source_cell_width: float,
    source_cell_height: float,
    source_file: str,
    source_id: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    if len(labels) != len(y_centers):
        raise ValueError("Supplemental labels and row centers must have equal length")
    subpaths, even_odd = black_ink_subpaths(page)
    scale_x = CELL_WIDTH / source_cell_width
    scale_y = CELL_HEIGHT / source_cell_height
    rows: list[dict[str, Any]] = []
    for row_index, (label, center_y) in enumerate(zip(labels, y_centers), start=1):
        slug = unicode_glyph_slug(label)
        row_dir = output_dir / slug
        row_dir.mkdir(parents=True, exist_ok=True)
        samples: list[dict[str, Any]] = []
        for sample_index, center_x in enumerate(x_centers, start=1):
            left = center_x - source_cell_width / 2
            top = center_y - source_cell_height / 2
            cell_paths = [
                subpath
                for subpath in subpaths
                if left <= subpath.center.x < left + source_cell_width
                and top <= subpath.center.y < top + source_cell_height
            ]
            if not cell_paths:
                continue
            svg_paths = [
                subpath_to_svg_path(
                    path,
                    left,
                    top,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                for path in cell_paths
            ]
            svg_path = row_dir / f"sample_{sample_index:02d}.svg"
            svg_path.write_text(_svg_document(svg_paths, even_odd), encoding="utf-8")
            bbox = fitz.Rect()
            for path in cell_paths:
                bbox.include_rect(path.bbox)
            canonical_bbox = [
                (bbox.x0 - left) * scale_x,
                (bbox.y0 - top) * scale_y,
                (bbox.x1 - left) * scale_x,
                (bbox.y1 - top) * scale_y,
            ]
            samples.append(
                {
                    "sample_index": sample_index,
                    "file": svg_path.relative_to(output_dir.parent).as_posix(),
                    "source_file": source_file,
                    "source_cell": {"center_x": center_x, "center_y": center_y},
                    "subpath_count": len(cell_paths),
                    "bbox": [round(value, 3) for value in canonical_bbox],
                }
            )
        rows.append(
            {
                "page": manifest_page,
                "row": row_index,
                "label": label,
                "slug": slug,
                "source_id": source_id,
                "mapping_status": "pending_user_confirmation",
                "sample_count": len(samples),
                "samples": samples,
            }
        )
    return rows


def append_english_symbols(source_pdf: Path, handwrite_dir: Path) -> dict[str, Any]:
    """Append English/Greek vector candidates without rebuilding Hebrew extraction."""
    handwrite_dir = handwrite_dir.resolve()
    source_pdf = source_pdf.resolve()
    extracted_dir = handwrite_dir / "extracted"
    manifest_path = handwrite_dir / "manifest.json"
    if not manifest_path.is_file() or not extracted_dir.is_dir():
        raise ValueError("Base handwriting extraction must exist before adding English samples")
    try:
        source_relative = source_pdf.relative_to(handwrite_dir).as_posix()
    except ValueError as exc:
        raise ValueError("Supplemental source must be stored inside HandWrite/") from exc

    with fitz.open(source_pdf) as document:
        if document.page_count != 2:
            raise ValueError(f"Expected two English sample pages, found {document.page_count}")
        rows = extract_supplemental_grid_page(
            document[0],
            manifest_page=4,
            labels=ENGLISH_PAGE_1_LABELS,
            x_centers=ENGLISH_X_CENTERS,
            y_centers=ENGLISH_PAGE_1_Y_CENTERS,
            source_cell_width=28.0,
            source_cell_height=28.0,
            source_file=source_relative,
            source_id=ENGLISH_SOURCE_ID,
            output_dir=extracted_dir,
        )
        rows.extend(
            extract_supplemental_grid_page(
                document[1],
                manifest_page=5,
                labels=SYMBOL_PAGE_2_LABELS,
                x_centers=SYMBOL_X_CENTERS,
                y_centers=SYMBOL_PAGE_2_Y_CENTERS,
                source_cell_width=28.0,
                source_cell_height=36.0,
                source_file=source_relative,
                source_id=ENGLISH_SOURCE_ID,
                output_dir=extracted_dir,
            )
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"] = [
        row for row in manifest["rows"] if row.get("source_id") != ENGLISH_SOURCE_ID
    ] + rows
    source_entry = {
        "file": source_relative,
        "sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
        "source_id": ENGLISH_SOURCE_ID,
        "mapping": {
            "page_1": "A-N, lowercase n, O-Z",
            "page_2": "Σ, Π, θ",
        },
        "status": "pending_user_confirmation",
    }
    supplemental = [
        item
        for item in manifest.get("supplemental_sources", [])
        if item.get("source_id") != ENGLISH_SOURCE_ID
    ]
    supplemental.append(source_entry)
    manifest["supplemental_sources"] = supplemental
    manifest["status"] = "vector_candidates_extracted_pending_english_review"
    manifest["extraction"]["row_count"] = len(manifest["rows"])
    manifest["extraction"]["sample_count"] = sum(
        row["sample_count"] for row in manifest["rows"]
    )
    manifest["extraction"]["pending_review_rows"] = [
        row["slug"]
        for row in manifest["rows"]
        if row.get("mapping_status") == "pending_user_confirmation"
    ]
    if "font_build" in manifest:
        manifest["font_build"]["pending_source_review"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    previews_dir = handwrite_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    preview_html = previews_dir / "english_glyph_contact_sheet.html"
    preview_html.write_text(build_contact_sheet(rows, handwrite_dir), encoding="utf-8")
    return {
        "source": source_entry,
        "row_count": len(rows),
        "sample_count": sum(row["sample_count"] for row in rows),
        "rows": rows,
        "preview_html": preview_html.relative_to(handwrite_dir).as_posix(),
    }


def confirm_english_symbols(
    handwrite_dir: Path,
    *,
    mirror_uppercase_to_lowercase: bool = True,
) -> dict[str, Any]:
    """Confirm the reviewed mappings and optionally reuse A-Z for a-z."""
    handwrite_dir = handwrite_dir.resolve()
    manifest_path = handwrite_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    other_rows = [
        row for row in manifest["rows"] if row.get("source_id") != ENGLISH_SOURCE_ID
    ]
    source_rows = [
        row
        for row in manifest["rows"]
        if row.get("source_id") == ENGLISH_SOURCE_ID and not row.get("case_alias")
    ]
    if not source_rows:
        raise ValueError("English supplemental rows have not been extracted")
    for row in source_rows:
        row["mapping_status"] = "user_confirmed"

    aliases: list[dict[str, Any]] = []
    existing_labels = {row["label"] for row in source_rows}
    if mirror_uppercase_to_lowercase:
        for row in source_rows:
            label = row["label"]
            if not (len(label) == 1 and "A" <= label <= "Z"):
                continue
            lowercase = label.lower()
            if lowercase in existing_labels:
                continue
            alias = copy.deepcopy(row)
            alias["label"] = lowercase
            alias["slug"] = unicode_glyph_slug(lowercase)
            alias["mapping_status"] = "user_confirmed_mirrored_case"
            alias["case_alias"] = True
            alias["mirrored_from"] = label
            aliases.append(alias)

    manifest["rows"] = other_rows + source_rows + aliases
    for source in manifest.get("supplemental_sources", []):
        if source.get("source_id") == ENGLISH_SOURCE_ID:
            source["status"] = "user_confirmed"
            source["mirror_uppercase_to_lowercase"] = mirror_uppercase_to_lowercase
            source["mapping"]["page_1"] = (
                "A-Z mirrored to a-z; dedicated lowercase n row overrides N"
                if mirror_uppercase_to_lowercase
                else "A-N, lowercase n, O-Z"
            )
    manifest["status"] = "english_mappings_confirmed_ready_for_font_build"
    manifest["extraction"]["row_count"] = len(manifest["rows"])
    manifest["extraction"]["sample_count"] = sum(
        row["sample_count"] for row in manifest["rows"]
    )
    manifest["extraction"]["pending_review_rows"] = []
    if "font_build" in manifest:
        manifest["font_build"]["pending_source_review"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "status": manifest["status"],
        "confirmed_source_rows": len(source_rows),
        "case_alias_rows": len(aliases),
        "dedicated_lowercase_n": "n" in existing_labels,
    }


def build_contact_sheet(rows: list[dict[str, Any]], root_dir: Path) -> str:
    row_markup: list[str] = []
    for row in rows:
        display_label = row["label"] or f"עמוד {row['page']} / שורה {row['row']} - מיפוי נדרש"
        cards = []
        for sample in row["samples"]:
            relative = Path(sample["file"])
            cards.append(
                '<figure class="sample">'
                f'<img src="../{relative.as_posix()}" alt="sample {sample["sample_index"]}">'
                f'<figcaption>{sample["sample_index"]}</figcaption>'
                "</figure>"
            )
        row_markup.append(
            '<section class="glyph-row">'
            f'<header><strong>{html.escape(display_label)}</strong> '
            f'<code>{html.escape(row["slug"])}</code> '
            f'<span>{row["sample_count"]} samples</span></header>'
            f'<div class="samples">{"".join(cards)}</div>'
            "</section>"
        )
    return f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><title>Handwriting extraction QA</title>
<style>
@page {{ size: A4 landscape; margin: 10mm; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: #111; background: #fff; font-family: Arial, sans-serif; }}
h1 {{ margin: 0 0 5mm; font-size: 22px; }}
.glyph-row {{ break-inside: avoid; border-bottom: 1px solid #bbb; padding: 2mm 0; }}
.glyph-row header {{ display: flex; gap: 8px; align-items: baseline; margin-bottom: 1mm; }}
.glyph-row header strong {{ min-width: 115px; font-size: 18px; }}
.glyph-row header span {{ color: #555; font-size: 11px; }}
.samples {{ direction: rtl; display: grid; grid-template-columns: repeat(11, 1fr); gap: 3px; }}
.sample {{ margin: 0; height: 18mm; border: 1px solid #d7d7d7; display: grid; grid-template-rows: 1fr auto; place-items: center; background: #fff; }}
.sample img {{ width: 9mm; height: 13mm; object-fit: contain; }}
.sample figcaption {{ font: 9px Consolas, monospace; color: #777; }}
</style></head><body>
<h1>בדיקת חילוץ כתב היד הווקטורי</h1>
{''.join(row_markup)}
</body></html>"""


def extract(source_pdf: Path, handwrite_dir: Path) -> dict[str, Any]:
    handwrite_dir = handwrite_dir.resolve()
    source_pdf = source_pdf.resolve()
    extracted_dir = (handwrite_dir / "extracted").resolve()
    if extracted_dir.parent != handwrite_dir or extracted_dir.name != "extracted":
        raise ValueError(f"Unsafe extraction directory: {extracted_dir}")
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True)
    previews_dir = handwrite_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(source_pdf) as document:
        if document.page_count != 2:
            raise ValueError(f"Expected two source pages, found {document.page_count}")
        page_count = document.page_count
        rows = extract_page(
            document[0], 1, PAGE_1_Y_CENTERS, HEBREW_LETTERS, extracted_dir
        )
        rows.extend(
            extract_page(document[1], 2, PAGE_2_Y_CENTERS, PAGE_2_LABELS, extracted_dir)
        )
    supplemental_path = handwrite_dir / "input" / SUPPLEMENTAL_FINAL_FILENAME
    supplemental_sources: list[dict[str, Any]] = []
    if supplemental_path.is_file():
        rows.extend(extract_supplemental_final_letters(supplemental_path, extracted_dir))
        supplemental_sources.append(
            {
                "file": supplemental_path.relative_to(handwrite_dir).as_posix(),
                "sha256": hashlib.sha256(supplemental_path.read_bytes()).hexdigest(),
                "mapping": {"row_1": "ן", "row_2": "ך"},
            }
        )
    source_hash = hashlib.sha256(source_pdf.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "status": "vector_candidates_extracted_pending_review",
        "source": {
            "file": source_pdf.relative_to(handwrite_dir).as_posix(),
            "sha256": source_hash,
            "page_count": page_count,
        },
        "supplemental_sources": supplemental_sources,
        "extraction": {
            "method": "pdf_filled_outline_subpaths",
            "raster_tracing_used": False,
            "cell_width_points": CELL_WIDTH,
            "cell_height_points": CELL_HEIGHT,
            "row_count": len(rows),
            "sample_count": sum(row["sample_count"] for row in rows),
            "unmapped_rows": [row["slug"] for row in rows if row["label"] is None],
        },
        "rows": rows,
        "activation": {
            "personal_font_active": False,
            "fallback": "Gveret Levin synthetic variants",
        },
    }
    manifest_path = handwrite_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    preview_html = previews_dir / "glyph_contact_sheet.html"
    preview_html.write_text(build_contact_sheet(rows, handwrite_dir), encoding="utf-8")
    return payload


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", type=Path)
    source_group.add_argument("--english-source", type=Path)
    source_group.add_argument("--confirm-english", action="store_true")
    parser.add_argument("--handwrite-dir", type=Path, default=Path("HandWrite"))
    args = parser.parse_args()
    if args.confirm_english:
        payload = confirm_english_symbols(args.handwrite_dir)
    elif args.english_source:
        payload = append_english_symbols(args.english_source, args.handwrite_dir)
    else:
        payload = extract(args.source, args.handwrite_dir)
        payload = payload["extraction"]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
