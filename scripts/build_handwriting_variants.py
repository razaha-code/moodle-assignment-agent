"""Build deterministic synthetic Hebrew handwriting variants.

Only the outlines mapped to Hebrew letters U+05D0..U+05EA are changed. The
advance width, cmap, punctuation, digits, Latin text, and mathematical glyphs
remain identical to the source font.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
from fontTools.ttLib.tables.ttProgram import Program


HEBREW_LETTER_CODEPOINTS = tuple(range(0x05D0, 0x05EB))
VARIANT_COUNT = 4
VARIANT_AFFINES = (
    (0.986, 1.010, -0.012, 5.5),
    (1.014, 0.992, 0.010, 6.5),
    (0.994, 1.014, 0.016, 7.0),
    (1.009, 0.987, -0.017, 7.5),
)


def _seed_for(glyph_name: str, variant_index: int) -> int:
    digest = hashlib.sha256(
        f"gveret-levin-synthetic-v1:{glyph_name}:{variant_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _mutate_simple_glyph(font: TTFont, glyph_name: str, variant_index: int) -> bool:
    glyf = font["glyf"]
    glyph = glyf[glyph_name]
    if glyph.isComposite() or glyph.numberOfContours <= 0:
        return False

    coordinates, _end_points, _flags = glyph.getCoordinates(glyf)
    if not coordinates:
        return False

    original = [(point[0], point[1]) for point in coordinates]
    min_x = min(x for x, _ in original)
    max_x = max(x for x, _ in original)
    min_y = min(y for _, y in original)
    max_y = max(y for _, y in original)
    width = max(max_x - min_x, 1)
    height = max(max_y - min_y, 1)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    scale_x, scale_y, shear, wave_amplitude = VARIANT_AFFINES[variant_index]
    rng = random.Random(_seed_for(glyph_name, variant_index))
    phase_x = rng.uniform(-math.pi, math.pi)
    phase_y = rng.uniform(-math.pi, math.pi)
    drift_x = rng.uniform(-2.5, 2.5)
    drift_y = rng.uniform(-2.0, 2.0)

    transformed: list[tuple[int, int]] = []
    for point_index, (x, y) in enumerate(original):
        nx = (x - center_x) / width
        ny = (y - center_y) / height
        local_phase = point_index * 0.31
        wave_x = wave_amplitude * math.sin((ny * math.pi * 1.7) + phase_x + local_phase)
        wave_y = wave_amplitude * 0.62 * math.sin((nx * math.pi * 2.1) + phase_y - local_phase)
        new_x = center_x + ((x - center_x) * scale_x) + shear * (y - center_y) + wave_x + drift_x
        new_y = center_y + ((y - center_y) * scale_y) + wave_y + drift_y
        transformed.append((round(new_x), round(new_y)))

    # Guard against an unexpectedly wild outline. A failed glyph remains original.
    if any(
        abs(x - ox) > 35
        or abs(y - oy) > 35
        or not -32768 <= x <= 32767
        or not -32768 <= y <= 32767
        for (x, y), (ox, oy) in zip(transformed, original)
    ):
        return False

    try:
        for point_index, (new_x, new_y) in enumerate(transformed):
            coordinates[point_index] = (new_x, new_y)
        glyph.coordinates = coordinates
        glyph.recalcBounds(glyf)
        if glyph.xMin >= glyph.xMax or glyph.yMin >= glyph.yMax:
            raise ValueError("mutated glyph has invalid bounds")
        glyph.program = Program()
        return True
    except Exception:
        glyph.coordinates = GlyphCoordinates(original)
        glyph.recalcBounds(glyf)
        return False


def _set_variant_names(font: TTFont, variant_number: int) -> None:
    family = f"Gveret Levin Synthetic {variant_number}"
    postscript_name = f"GveretLevin-Synthetic-{variant_number}"
    name_table = font["name"]
    for name_id, value in ((1, family), (2, "Regular"), (4, family), (6, postscript_name)):
        name_table.setName(value, name_id, 3, 1, 0x409)
        name_table.setName(value, name_id, 1, 0, 0)


def build_variants(source_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    built: list[Path] = []
    for variant_index in range(VARIANT_COUNT):
        font = TTFont(source_path, recalcTimestamp=False)
        cmap = font.getBestCmap()
        for codepoint in HEBREW_LETTER_CODEPOINTS:
            glyph_name = cmap.get(codepoint)
            if glyph_name:
                _mutate_simple_glyph(font, glyph_name, variant_index)
        _set_variant_names(font, variant_index + 1)
        font["head"].checkSumAdjustment = 0
        font.flavor = "woff2"
        output_path = output_dir / f"GveretLevin-Synthetic-{variant_index + 1}.woff2"
        font.save(output_path, reorderTables=True)
        built.append(output_path)
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in build_variants(args.source, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
