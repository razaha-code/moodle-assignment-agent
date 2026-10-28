# Personal handwriting pipeline

This directory contains only the assets and tooling used to build the personal
handwriting font. It is intentionally separated from the Moodle workflow and
from the synthetic Gveret Levin fallback.

## Layout

- `input/` - immutable source exports from the handwriting application.
- `extracted/` - automatically segmented vector glyph candidates.
- `reviewed/` - glyphs accepted or corrected after visual review.
- `fonts/` - generated personal TTF and WOFF2 font assets.
- `previews/` - raster previews and comparison documents used for QA.
- `scripts/` - extraction, normalization, and font-building tools.
- `manifest.json` - source identity, extraction state, and glyph mapping.

The active renderer uses the personal font by default. The previous Gveret
Levin variants remain selectable with the `--handwriting-font gveret` flag.
New supplemental rows stay excluded from font builds until their manifest
status is user-confirmed. Source PDFs must never be modified in place.

## Supplemental math symbols, 2026-10-27

`supplemental/math_symbols_2026_10_27/` contains 99 candidates in 11 rows
extracted from `Note 27 Oct 2026.pdf`. See `preview.html` or `preview.png`.
Each sample has an exact `.strokes.svg` and a filled-outline `.svg` that can be
read by `svg_to_glyph`. The stroke expansion uses round caps/joins and a 0.015pt
curve tolerance; the exact source strokes are retained for future refinement.
`manifest.json` records source SHA-256, row/sample identity, source bounds,
proposed labels, and the pending mapping status. `validation.json` records the
outline/source comparison and font-builder import check.

This collection is separate from the active manifest: it does not replace
existing glyphs. Before activation, confirm each row's Unicode mapping (especially
the first three rows and the final vertical stroke), select samples, and import
the selected rows into the active manifest with paths relative to `HandWrite/`.
Preserve the source bounds used for advance-width calculation. Then rebuild the
fonts and check baseline/size alongside existing letters. No assignment rerun is
needed for extraction or font building.

`WORD_SPACE_SCALE` in `scripts/build_personal_fonts.py` is now `1.25`.
The four active TTF/WOFF2 variants have been rebuilt with this value; existing
HTML/PDF artifacts keep their previously embedded font until explicitly rebuilt.
