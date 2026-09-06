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
