import tempfile
import unittest
from pathlib import Path

import fitz

from HandWrite.scripts.build_personal_fonts import svg_to_glyph
from HandWrite.scripts.extract_math_symbols import stroke_paths


class MathSymbolExtractionTests(unittest.TestCase):
    def test_open_round_stroke_has_filled_font_importable_outline(self):
        drawing = {'items': [('l', fitz.Point(3, 9), fitz.Point(12, 9))],
                   'width': 0.8, 'closePath': False}
        centerline, outlines = stroke_paths(drawing, 0, 0)
        self.assertNotIn('Z', centerline)
        self.assertEqual(3, len(outlines))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'minus.svg'
            path.write_text('<svg xmlns="http://www.w3.org/2000/svg">' +
                            ''.join(f'<path d="{p}"/>' for p in outlines) + '</svg>')
            glyph = svg_to_glyph(path, 600)
        self.assertEqual(3, glyph.numberOfContours)

    def test_cubic_source_is_preserved_and_not_rasterized(self):
        drawing = {'items': [('c', fitz.Point(0, 0), fitz.Point(0, 5),
                              fitz.Point(5, 5), fitz.Point(5, 0))],
                   'width': 0.8, 'closePath': False}
        centerline, outlines = stroke_paths(drawing, 0, 0)
        self.assertIn('C 0.0 5.0 5.0 5.0 5.0 0.0', centerline)
        self.assertGreater(len(outlines), 3)

    def test_unsupported_closed_stroke_is_not_silently_changed(self):
        drawing = {'items': [('l', fitz.Point(0, 0), fitz.Point(1, 1))],
                   'width': 0.8, 'closePath': True}
        with self.assertRaises(ValueError):
            stroke_paths(drawing, 0, 0)
