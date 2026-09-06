"""Build four personal handwriting fonts from reviewed vector candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.ttProgram import Program


VARIANT_COUNT = 4
SELECTION_FRACTIONS = (0.15, 0.38, 0.62, 0.85)
SVG_BASELINE = 23.5
FONT_SCALE = 50.0
# A narrow but non-zero margin on each side. This keeps letters inside a word
# close together while preserving the normal width of the space glyph.
SIDE_BEARING = 22
# Keep letters tight inside words, but leave a slightly clearer visual break
# between adjacent words than the source font provides.
WORD_SPACE_SCALE = 1.18
PERSONAL_ALTERNATE_SUFFIXES = (".ss01", ".ss02")
OPTIONAL_FALLBACK_CODEPOINTS = (0x05DA, 0x05DF)  # final kaf, final nun
TOKEN_RE = re.compile(r"[MLCZ]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _set_font_names(font: TTFont, variant_number: int) -> None:
    family = f"Raz Personal Handwriting {variant_number}"
    postscript_name = f"RazPersonalHandwriting-{variant_number}"
    for name_id, value in ((1, family), (2, "Regular"), (4, family), (6, postscript_name)):
        font["name"].setName(value, name_id, 3, 1, 0x409)
        font["name"].setName(value, name_id, 1, 0, 0)


def _selected_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(samples) < VARIANT_COUNT:
        raise ValueError(f"At least {VARIANT_COUNT} samples are required, found {len(samples)}")
    ordered = sorted(samples, key=lambda sample: sample["sample_index"])
    selected: list[dict[str, Any]] = []
    used_positions: set[int] = set()
    for fraction in SELECTION_FRACTIONS:
        position = round((len(ordered) - 1) * fraction)
        while position in used_positions and position + 1 < len(ordered):
            position += 1
        while position in used_positions and position > 0:
            position -= 1
        used_positions.add(position)
        selected.append(ordered[position])
    return selected


def _svg_paths(svg_path: Path) -> list[str]:
    root = ET.parse(svg_path).getroot()
    paths = [element.attrib["d"] for element in root.iter() if element.tag.endswith("path")]
    if not paths:
        raise ValueError(f"SVG contains no vector paths: {svg_path}")
    return paths


def _path_coordinates(path_data: str) -> list[tuple[float, float]]:
    tokens = TOKEN_RE.findall(path_data)
    coordinates: list[tuple[float, float]] = []
    index = 0
    while index < len(tokens):
        command = tokens[index]
        index += 1
        if command in {"M", "L"}:
            coordinates.append((float(tokens[index]), float(tokens[index + 1])))
            index += 2
        elif command == "C":
            coordinates.extend(
                (
                    (float(tokens[index]), float(tokens[index + 1])),
                    (float(tokens[index + 2]), float(tokens[index + 3])),
                    (float(tokens[index + 4]), float(tokens[index + 5])),
                )
            )
            index += 6
        elif command == "Z":
            continue
        else:
            raise ValueError(f"Unsupported SVG command: {command}")
    return coordinates


def _draw_svg_path(
    pen: Cu2QuPen,
    path_data: str,
    center_x: float,
    advance_width: int,
) -> None:
    tokens = TOKEN_RE.findall(path_data)
    index = 0

    def point_at(token_index: int) -> tuple[float, float]:
        x = float(tokens[token_index])
        y = float(tokens[token_index + 1])
        return (
            (x - center_x) * FONT_SCALE + advance_width / 2,
            (SVG_BASELINE - y) * FONT_SCALE,
        )

    while index < len(tokens):
        command = tokens[index]
        index += 1
        if command == "M":
            pen.moveTo(point_at(index))
            index += 2
        elif command == "L":
            pen.lineTo(point_at(index))
            index += 2
        elif command == "C":
            pen.curveTo(point_at(index), point_at(index + 2), point_at(index + 4))
            index += 6
        elif command == "Z":
            pen.closePath()
        else:
            raise ValueError(f"Unsupported SVG command: {command}")


def svg_to_glyph(svg_path: Path, advance_width: int):
    path_data = _svg_paths(svg_path)
    coordinates = [point for path in path_data for point in _path_coordinates(path)]
    min_x = min(point[0] for point in coordinates)
    max_x = max(point[0] for point in coordinates)
    center_x = (min_x + max_x) / 2
    tt_pen = TTGlyphPen(None)
    cubic_pen = Cu2QuPen(tt_pen, max_err=1.0, reverse_direction=False)
    for path in path_data:
        _draw_svg_path(cubic_pen, path, center_x, advance_width)
    glyph = tt_pen.glyph()
    glyph.program = Program()
    return glyph


def _advance_width(selected: list[dict[str, Any]]) -> int:
    widest = max(sample["bbox"][2] - sample["bbox"][0] for sample in selected)
    return max(220, min(920, round(widest * FONT_SCALE + SIDE_BEARING * 2)))


def _save_font_pair(font: TTFont, output_dir: Path, variant_number: int) -> tuple[Path, Path]:
    ttf_path = output_dir / f"RazPersonalHandwriting-{variant_number}.ttf"
    woff2_path = output_dir / f"RazPersonalHandwriting-{variant_number}.woff2"
    font["head"].checkSumAdjustment = 0
    font.flavor = None
    font.save(ttf_path, reorderTables=True)
    font.flavor = "woff2"
    font.save(woff2_path, reorderTables=True)
    return ttf_path, woff2_path


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def buildable_manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in manifest["rows"]
        if row.get("label")
        and row.get("mapping_status") != "pending_user_confirmation"
    ]


def ensure_unicode_glyph(font: TTFont, codepoint: int) -> tuple[str, bool]:
    cmap = font.getBestCmap()
    glyph_name = cmap.get(codepoint)
    if glyph_name is not None:
        return glyph_name, False
    glyph_name = f"uni{codepoint:04X}"
    glyph_order = font.getGlyphOrder()
    if glyph_name not in glyph_order:
        font.setGlyphOrder([*glyph_order, glyph_name])
        font["maxp"].numGlyphs = len(font.getGlyphOrder())
    for table in font["cmap"].tables:
        if table.isUnicode():
            table.cmap[codepoint] = glyph_name
    return glyph_name, True


def build_personal_fonts(
    handwrite_dir: Path,
    base_font_paths: tuple[Path, ...],
    output_dir: Path | None = None,
    *,
    update_manifest: bool = True,
) -> dict[str, Any]:
    handwrite_dir = handwrite_dir.resolve()
    output_dir = (output_dir or handwrite_dir / "fonts").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(base_font_paths) != VARIANT_COUNT:
        raise ValueError(f"Expected {VARIANT_COUNT} base fonts")
    manifest_path = handwrite_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapped_rows = buildable_manifest_rows(manifest)
    if manifest["extraction"].get("unmapped_rows"):
        raise ValueError("Cannot build fonts while glyph rows remain unmapped")

    selections: dict[int, list[dict[str, Any]]] = {}
    advances: dict[int, int] = {}
    for row in mapped_rows:
        label = row["label"]
        if len(label) != 1:
            continue
        codepoint = ord(label)
        selected = _selected_samples(row["samples"])
        selections[codepoint] = selected
        advances[codepoint] = _advance_width(selected)

    font_files: list[dict[str, Any]] = []
    space_advances: list[int] = []
    for variant_index, base_path in enumerate(base_font_paths):
        font = TTFont(base_path, recalcTimestamp=False)
        cmap = font.getBestCmap()
        glyf = font["glyf"]
        replaced: list[str] = []
        added: list[str] = []
        synchronized_alternates: list[str] = []
        for codepoint, selected in sorted(selections.items()):
            glyph_name, was_added = ensure_unicode_glyph(font, codepoint)
            sample = selected[variant_index]
            svg_path = handwrite_dir / sample["file"]
            glyph = svg_to_glyph(svg_path, advances[codepoint])
            glyf[glyph_name] = glyph
            glyph.recalcBounds(glyf)
            font["hmtx"].metrics[glyph_name] = (advances[codepoint], glyph.xMin)
            replaced.append(f"U+{codepoint:04X}")
            if was_added:
                added.append(f"U+{codepoint:04X}")
            # Gveret Levin contains contextual substitutions (notably ת+נ)
            # that can select inherited .ss01/.ss02 glyphs. Keep those targets
            # personal too, otherwise a word can unexpectedly render one base-
            # font letter among the handwritten glyphs.
            for suffix in PERSONAL_ALTERNATE_SUFFIXES:
                alternate_name = f"{glyph_name}{suffix}"
                if alternate_name not in glyf:
                    continue
                alternate_glyph = copy.deepcopy(glyph)
                glyf[alternate_name] = alternate_glyph
                alternate_glyph.recalcBounds(glyf)
                font["hmtx"].metrics[alternate_name] = (
                    advances[codepoint],
                    alternate_glyph.xMin,
                )
                synchronized_alternates.append(alternate_name)
        space_glyph_name = cmap.get(0x20)
        if space_glyph_name is not None:
            original_space_advance, space_lsb = font["hmtx"].metrics[space_glyph_name]
            enlarged_space_advance = round(original_space_advance * WORD_SPACE_SCALE)
            font["hmtx"].metrics[space_glyph_name] = (enlarged_space_advance, space_lsb)
            space_advances.append(enlarged_space_advance)
        _set_font_names(font, variant_index + 1)
        ttf_path, woff2_path = _save_font_pair(font, output_dir, variant_index + 1)
        font_files.append(
            {
                "variant": variant_index + 1,
                "ttf": _portable_path(ttf_path, handwrite_dir),
                "woff2": _portable_path(woff2_path, handwrite_dir),
                "replaced_codepoints": replaced,
                "added_codepoints": added,
                "synchronized_alternates": synchronized_alternates,
                "sha256_ttf": hashlib.sha256(ttf_path.read_bytes()).hexdigest(),
                "sha256_woff2": hashlib.sha256(woff2_path.read_bytes()).hexdigest(),
            }
        )

    report = {
        "schema_version": 1,
        "status": "built_pending_visual_approval",
        "variant_count": VARIANT_COUNT,
        "font_scale": FONT_SCALE,
        "svg_baseline": SVG_BASELINE,
        "side_bearing": SIDE_BEARING,
        "word_space_scale": WORD_SPACE_SCALE,
        "space_advances": space_advances,
        "selection_fractions": list(SELECTION_FRACTIONS),
        "mapped_codepoints": [f"U+{codepoint:04X}" for codepoint in sorted(selections)],
        "fallback_codepoints": [
            f"U+{codepoint:04X}"
            for codepoint in OPTIONAL_FALLBACK_CODEPOINTS
            if codepoint not in selections
        ],
        "selections": {
            f"U+{codepoint:04X}": [sample["sample_index"] for sample in selected]
            for codepoint, selected in sorted(selections.items())
        },
        "advances": {f"U+{codepoint:04X}": width for codepoint, width in sorted(advances.items())},
        "fonts": font_files,
    }
    report_path = output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if update_manifest:
        manifest["font_build"] = {
            "status": report["status"],
            "report": _portable_path(report_path, handwrite_dir),
            "variant_count": VARIANT_COUNT,
            "fallback_codepoints": report["fallback_codepoints"],
        }
        manifest["activation"] = {
            "personal_font_active": True,
            "default": "personal",
            "selectable_alternative": "Gveret Levin synthetic variants",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handwrite-dir", type=Path, default=Path("HandWrite"))
    parser.add_argument("--base-font-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    base_paths = tuple(
        args.base_font_dir / f"GveretLevin-Synthetic-{index}.woff2"
        for index in range(1, VARIANT_COUNT + 1)
    )
    report = build_personal_fonts(args.handwrite_dir, base_paths, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
