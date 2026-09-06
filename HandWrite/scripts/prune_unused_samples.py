"""Remove extracted SVG candidates that are not used by the active font variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def prune_unused_samples(handwrite_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Keep the four samples selected in the last successful font build.

    Only ``*.svg`` files below ``HandWrite/extracted`` are eligible for removal.
    Source PDFs, generated fonts and preview artifacts are never touched.
    """

    handwrite_dir = handwrite_dir.resolve()
    manifest_path = handwrite_dir / "manifest.json"
    report_path = handwrite_dir / "fonts" / "build_report.json"
    extracted_dir = (handwrite_dir / "extracted").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selections = report.get("selections", {})

    kept_files: set[Path] = set()
    changed_rows = 0
    removed_references = 0
    for row in manifest.get("rows", []):
        label = row.get("label")
        samples = row.get("samples", [])
        if not label or len(label) != 1 or not samples:
            continue
        selected_indices = set(selections.get(f"U+{ord(label):04X}", []))
        if not selected_indices:
            raise ValueError(f"No active font selection recorded for {label!r}")
        selected_samples = [
            sample for sample in samples if sample.get("sample_index") in selected_indices
        ]
        if len(selected_samples) != len(selected_indices):
            raise ValueError(f"Missing selected SVG sample for {label!r}")
        if len(selected_samples) != len(samples):
            changed_rows += 1
            removed_references += len(samples) - len(selected_samples)
        row["samples"] = selected_samples
        row["sample_count"] = len(selected_samples)
        for sample in selected_samples:
            sample_path = (handwrite_dir / sample["file"]).resolve()
            if not _inside(sample_path, extracted_dir):
                raise ValueError(f"Sample path escapes extracted directory: {sample_path}")
            kept_files.add(sample_path)

    all_svg_files = {path.resolve() for path in extracted_dir.rglob("*.svg")}
    unused_files = sorted(all_svg_files - kept_files, key=str)
    for path in unused_files:
        if not _inside(path, extracted_dir):
            raise ValueError(f"Refusing to remove file outside extracted directory: {path}")

    result = {
        "dry_run": dry_run,
        "svg_files_before": len(all_svg_files),
        "svg_files_kept": len(kept_files),
        "svg_files_removed": len(unused_files),
        "manifest_rows_changed": changed_rows,
        "manifest_references_removed": removed_references,
        "recoverable_from_source_pdfs": True,
    }
    if dry_run:
        return result

    for path in unused_files:
        path.unlink()
    manifest["extraction"]["sample_count"] = sum(
        len(row.get("samples", [])) for row in manifest.get("rows", [])
    )
    manifest["sample_cleanup"] = result
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handwrite-dir", type=Path, default=Path("HandWrite"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prune_unused_samples(args.handwrite_dir, dry_run=args.dry_run),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
