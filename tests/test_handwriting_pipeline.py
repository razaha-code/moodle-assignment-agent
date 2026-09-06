import hashlib
import tempfile
import unittest
from pathlib import Path

import fitz

from HandWrite.scripts.extract_vector_glyphs import (
    HEBREW_LETTERS,
    black_ink_subpaths,
    extract,
)


class HandwritingPipelineTests(unittest.TestCase):
    source_path = Path("HandWrite/input/handwriting_samples_2026-08-21.pdf")
    supplemental_path = Path("HandWrite/input/final_kaf_nun_samples_2026-08-21.pdf")

    def test_source_contains_filled_vector_ink(self):
        document = fitz.open(self.source_path)
        for page in document:
            paths, _ = black_ink_subpaths(page)
            self.assertGreater(len(paths), 100)

    def test_extraction_is_deterministic_and_preserves_source(self):
        original_hash = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_root = Path(first_dir) / "HandWrite"
            second_root = Path(second_dir) / "HandWrite"
            (first_root / "input").mkdir(parents=True)
            (second_root / "input").mkdir(parents=True)
            first_source = first_root / "input" / self.source_path.name
            second_source = second_root / "input" / self.source_path.name
            first_source.write_bytes(self.source_path.read_bytes())
            second_source.write_bytes(self.source_path.read_bytes())
            first = extract(first_source, first_root)
            second = extract(second_source, second_root)
            first_svgs = sorted((first_root / "extracted").rglob("*.svg"))
            second_svgs = sorted((second_root / "extracted").rglob("*.svg"))
            self.assertEqual(first["extraction"], second["extraction"])
            self.assertEqual(len(first_svgs), len(second_svgs))
            self.assertEqual(
                [hashlib.sha256(path.read_bytes()).digest() for path in first_svgs],
                [hashlib.sha256(path.read_bytes()).digest() for path in second_svgs],
            )
        self.assertEqual(original_hash, hashlib.sha256(self.source_path.read_bytes()).hexdigest())

    def test_regular_hebrew_rows_are_mapped_in_alphabetical_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "HandWrite"
            (root / "input").mkdir(parents=True)
            source = root / "input" / self.source_path.name
            source.write_bytes(self.source_path.read_bytes())
            payload = extract(source, root)
        page_one = [row for row in payload["rows"] if row["page"] == 1]
        self.assertEqual(list(HEBREW_LETTERS), [row["label"] for row in page_one])
        self.assertTrue(all(row["sample_count"] >= 8 for row in page_one))

    def test_supplemental_final_kaf_and_nun_are_extracted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "HandWrite"
            (root / "input").mkdir(parents=True)
            source = root / "input" / self.source_path.name
            supplement = root / "input" / self.supplemental_path.name
            source.write_bytes(self.source_path.read_bytes())
            supplement.write_bytes(self.supplemental_path.read_bytes())
            payload = extract(source, root)
        supplemental_rows = [row for row in payload["rows"] if row["page"] == 3]
        self.assertEqual(["ן", "ך"], [row["label"] for row in supplemental_rows])
        self.assertEqual([10, 10], [row["sample_count"] for row in supplemental_rows])
        self.assertTrue(
            all(row["mapping_status"] == "user_confirmed_supplemental" for row in supplemental_rows)
        )

    def test_user_confirmed_final_letter_rows_are_mapped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "HandWrite"
            (root / "input").mkdir(parents=True)
            source = root / "input" / self.source_path.name
            source.write_bytes(self.source_path.read_bytes())
            payload = extract(source, root)
        final_rows = [
            row for row in payload["rows"] if row["page"] == 2 and row["row"] <= 3
        ]
        self.assertEqual(["ץ", "ף", "ם"], [row["label"] for row in final_rows])
        self.assertTrue(all(row["mapping_status"] == "user_confirmed" for row in final_rows))
        self.assertEqual([], payload["extraction"]["unmapped_rows"])


if __name__ == "__main__":
    unittest.main()
