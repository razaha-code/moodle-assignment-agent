import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fontTools.ttLib import TTFont

from HandWrite.scripts.build_personal_fonts import (
    SIDE_BEARING,
    WORD_SPACE_SCALE,
    buildable_manifest_rows,
    build_personal_fonts,
)
from HandWrite.scripts.extract_vector_glyphs import (
    ENGLISH_PAGE_1_LABELS,
    ENGLISH_PAGE_1_Y_CENTERS,
    SYMBOL_PAGE_2_LABELS,
    SYMBOL_PAGE_2_Y_CENTERS,
    confirm_english_symbols,
)
from HandWrite.scripts.prune_unused_samples import prune_unused_samples


class PersonalHandwritingFontTests(unittest.TestCase):
    handwrite_dir = Path("HandWrite")
    base_paths = tuple(
        Path("assets/fonts/gveret_levin") / f"GveretLevin-Synthetic-{index}.woff2"
        for index in range(1, 5)
    )

    def test_english_supplement_layout_has_one_row_per_label(self):
        self.assertEqual(len(ENGLISH_PAGE_1_LABELS), len(ENGLISH_PAGE_1_Y_CENTERS))
        self.assertEqual(tuple("ABCDEFGHIJKLMN") + ("n",) + tuple("OPQRSTUVWXYZ"), ENGLISH_PAGE_1_LABELS)
        self.assertEqual(len(SYMBOL_PAGE_2_LABELS), len(SYMBOL_PAGE_2_Y_CENTERS))
        self.assertEqual(("Σ", "Π", "θ"), SYMBOL_PAGE_2_LABELS)

    def test_pending_user_rows_are_not_built_into_active_fonts(self):
        manifest = {
            "rows": [
                {"label": "א", "mapping_status": "confirmed_layout"},
                {"label": "A", "mapping_status": "pending_user_confirmation"},
                {"label": None, "mapping_status": "needs_user_review"},
            ]
        }

        self.assertEqual(
            [{"label": "א", "mapping_status": "confirmed_layout"}],
            buildable_manifest_rows(manifest),
        )

    def test_builds_four_fonts_with_equal_variant_widths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fonts"
            report = build_personal_fonts(
                self.handwrite_dir, self.base_paths, output_dir, update_manifest=False
            )
            fonts = [TTFont(output_dir / f"RazPersonalHandwriting-{index}.woff2") for index in range(1, 5)]
            for codepoint_label in report["mapped_codepoints"]:
                codepoint = int(codepoint_label[2:], 16)
                glyph_names = [font.getBestCmap()[codepoint] for font in fonts]
                widths = [font["hmtx"].metrics[name][0] for font, name in zip(fonts, glyph_names)]
                self.assertEqual(1, len(set(widths)))
            self.assertEqual([], report["fallback_codepoints"])
            self.assertEqual(22, SIDE_BEARING)
            self.assertEqual(SIDE_BEARING, report["side_bearing"])
            self.assertEqual(1.18, WORD_SPACE_SCALE)
            self.assertEqual(WORD_SPACE_SCALE, report["word_space_scale"])
            self.assertEqual(1, len(set(report["space_advances"])))
            base_font = TTFont(self.base_paths[0])
            base_space_name = base_font.getBestCmap()[0x20]
            base_space_advance = base_font["hmtx"].metrics[base_space_name][0]
            self.assertEqual(round(base_space_advance * WORD_SPACE_SCALE), report["space_advances"][0])

    def test_personal_hebrew_outlines_vary_across_fonts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fonts"
            build_personal_fonts(
                self.handwrite_dir, self.base_paths, output_dir, update_manifest=False
            )
            fonts = [TTFont(output_dir / f"RazPersonalHandwriting-{index}.woff2") for index in range(1, 5)]
            for codepoint in (0x05D0, 0x05DC, 0x05E9, 0x05E5, 0x05E3, 0x05DD):
                outlines = []
                for font in fonts:
                    glyph_name = font.getBestCmap()[codepoint]
                    coordinates, _, _ = font["glyf"][glyph_name].getCoordinates(font["glyf"])
                    outlines.append(tuple(coordinates))
                self.assertGreaterEqual(len(set(outlines)), 3)

    def test_personal_fonts_include_latin_and_confirmed_greek_glyphs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fonts"
            report = build_personal_fonts(
                self.handwrite_dir, self.base_paths, output_dir, update_manifest=False
            )
            expected = {ord("A"), ord("Z"), ord("a"), ord("n"), ord("z"), ord("Σ"), ord("Π"), ord("θ")}
            self.assertTrue(expected.issubset({int(item[2:], 16) for item in report["mapped_codepoints"]}))
            fonts = [TTFont(output_dir / f"RazPersonalHandwriting-{index}.woff2") for index in range(1, 5)]
            for font in fonts:
                self.assertTrue(expected.issubset(font.getBestCmap()))
                upper_a = font.getBestCmap()[ord("A")]
                lower_a = font.getBestCmap()[ord("a")]
                upper_coordinates, _, _ = font["glyf"][upper_a].getCoordinates(font["glyf"])
                lower_coordinates, _, _ = font["glyf"][lower_a].getCoordinates(font["glyf"])
                self.assertEqual(tuple(upper_coordinates), tuple(lower_coordinates))
            sigma_outlines = []
            for font in fonts:
                sigma_name = font.getBestCmap()[ord("Σ")]
                coordinates, _, _ = font["glyf"][sigma_name].getCoordinates(font["glyf"])
                sigma_outlines.append(tuple(coordinates))
            self.assertGreaterEqual(len(set(sigma_outlines)), 3)

    def test_confirm_english_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "HandWrite"
            root.mkdir()
            (root / "manifest.json").write_text(
                (self.handwrite_dir / "manifest.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            first = confirm_english_symbols(root)
            second = confirm_english_symbols(root)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            aliases = [row for row in manifest["rows"] if row.get("case_alias")]

            self.assertEqual(25, first["case_alias_rows"])
            self.assertEqual(25, second["case_alias_rows"])
            self.assertEqual(25, len(aliases))

    def test_contextual_nun_alternates_use_the_personal_outline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fonts"
            build_personal_fonts(
                self.handwrite_dir, self.base_paths, output_dir, update_manifest=False
            )
            for index in range(1, 5):
                font = TTFont(output_dir / f"RazPersonalHandwriting-{index}.woff2")
                glyph_name = font.getBestCmap()[0x05E0]
                coordinates, _, _ = font["glyf"][glyph_name].getCoordinates(font["glyf"])
                metrics = font["hmtx"].metrics[glyph_name]
                for suffix in (".ss01", ".ss02"):
                    alternate_name = f"{glyph_name}{suffix}"
                    alternate_coordinates, _, _ = font["glyf"][alternate_name].getCoordinates(
                        font["glyf"]
                    )
                    self.assertEqual(tuple(coordinates), tuple(alternate_coordinates))
                    self.assertEqual(metrics, font["hmtx"].metrics[alternate_name])

    def test_font_build_is_binary_deterministic(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_output = Path(first_dir) / "fonts"
            second_output = Path(second_dir) / "fonts"
            build_personal_fonts(
                self.handwrite_dir, self.base_paths, first_output, update_manifest=False
            )
            build_personal_fonts(
                self.handwrite_dir, self.base_paths, second_output, update_manifest=False
            )
            for index in range(1, 5):
                first = first_output / f"RazPersonalHandwriting-{index}.woff2"
                second = second_output / f"RazPersonalHandwriting-{index}.woff2"
                self.assertEqual(
                    hashlib.sha256(first.read_bytes()).digest(),
                    hashlib.sha256(second.read_bytes()).digest(),
                )

    def test_prune_unused_samples_only_removes_unselected_extracted_svgs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "HandWrite"
            (root / "fonts").mkdir(parents=True)
            (root / "extracted" / "u05d0").mkdir(parents=True)
            samples = []
            for index in range(1, 6):
                relative = f"extracted/u05d0/sample_{index:02d}.svg"
                (root / relative).write_text("<svg/>", encoding="utf-8")
                samples.append({"sample_index": index, "file": relative})
            manifest = {
                "extraction": {"sample_count": 5},
                "rows": [{"label": "א", "sample_count": 5, "samples": samples}],
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
            (root / "fonts" / "build_report.json").write_text(
                json.dumps({"selections": {"U+05D0": [1, 2, 4, 5]}}), encoding="utf-8"
            )

            preview = prune_unused_samples(root, dry_run=True)
            self.assertEqual(1, preview["svg_files_removed"])
            self.assertTrue((root / "extracted/u05d0/sample_03.svg").exists())

            result = prune_unused_samples(root)
            self.assertEqual(1, result["svg_files_removed"])
            self.assertFalse((root / "extracted/u05d0/sample_03.svg").exists())
            self.assertTrue((root / "extracted/u05d0/sample_04.svg").exists())
            updated = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(4, updated["rows"][0]["sample_count"])
            self.assertEqual(4, updated["extraction"]["sample_count"])


if __name__ == "__main__":
    unittest.main()
