import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import (
    CompoundReference,
    DocumentManifest,
    ManifestQuestion,
    VisualEvidence,
)


MANIFEST_SCHEMA_VERSION = 3


def compute_source_hash(saved_files: list[dict[str, Any]], attachment_texts: list[str]) -> str:
    digest = hashlib.sha256()
    for item in sorted(saved_files, key=lambda value: str(value.get("local_path", ""))):
        path = Path(str(item.get("local_path", "")))
        digest.update(str(item.get("filename", path.name)).encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    for text in attachment_texts:
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _compound_to_dict(compound: CompoundReference) -> dict[str, str]:
    return {
        "iupac_name": compound.iupac_name,
        "smiles": compound.smiles,
        "role": compound.role,
    }


def manifest_to_dict(manifest: DocumentManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "source_hash": manifest.source_hash,
        "model": manifest.model,
        "assignment_id": manifest.assignment_id,
        "suggested_topic": manifest.suggested_topic,
        "created_at": manifest.created_at,
        "vision_request_count": manifest.vision_request_count,
        "fallback_crop_count": manifest.fallback_crop_count,
        "manifest_retry_count": manifest.manifest_retry_count,
        "retried_pages": list(manifest.retried_pages),
        "questions": [
            {
                "id": question.id,
                "label": question.label,
                "top_level": question.top_level,
                "page_indices": list(question.page_indices),
                "prompt_text": question.prompt_text,
                "prompt_summary": question.prompt_summary,
                "visual": {
                    "description": question.visual.description,
                    "reactants": list(question.visual.reactants),
                    "reagents": list(question.visual.reagents),
                    "conditions": list(question.visual.conditions),
                    "stereochemistry": question.visual.stereochemistry,
                    "compounds": [_compound_to_dict(value) for value in question.visual.compounds],
                    "bounding_box": list(question.visual.bounding_box) if question.visual.bounding_box else None,
                    "confidence": question.visual.confidence,
                    "unreadable": question.visual.unreadable,
                    "fallback_used": question.visual.fallback_used,
                },
            }
            for question in manifest.questions
        ],
    }


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def question_from_dict(raw: dict[str, Any], default_pages: tuple[int, ...] = ()) -> ManifestQuestion | None:
    item_id = str(raw.get("id") or "").strip()
    if not item_id:
        return None
    raw_pages = raw.get("page_indices")
    pages = tuple(
        int(page) for page in raw_pages
        if isinstance(page, int) and page > 0
    ) if isinstance(raw_pages, list) else default_pages
    visual_raw = raw.get("visual") or {}
    if not isinstance(visual_raw, dict):
        visual_raw = {}
    raw_box = visual_raw.get("bounding_box")
    bounding_box = None
    if isinstance(raw_box, list) and len(raw_box) == 4:
        try:
            values = tuple(max(0.0, min(1.0, float(value))) for value in raw_box)
            if values[2] > values[0] and values[3] > values[1]:
                bounding_box = values
        except (TypeError, ValueError):
            pass
    compounds: list[CompoundReference] = []
    for compound in visual_raw.get("compounds") or []:
        if isinstance(compound, dict):
            compounds.append(
                CompoundReference(
                    iupac_name=str(compound.get("iupac_name") or "").strip(),
                    smiles=str(compound.get("smiles") or "").strip(),
                    role=str(compound.get("role") or "").strip(),
                )
            )
    try:
        confidence = max(0.0, min(1.0, float(visual_raw.get("confidence", 1.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return ManifestQuestion(
        id=item_id,
        label=str(raw.get("label") or item_id).strip(),
        top_level=str(raw.get("top_level") or item_id).strip(),
        page_indices=pages or default_pages,
        prompt_text=str(raw.get("prompt_text") or raw.get("prompt_summary") or "").strip(),
        prompt_summary=str(raw.get("prompt_summary") or raw.get("prompt_text") or "").strip(),
        visual=VisualEvidence(
            description=str(visual_raw.get("description") or "").strip(),
            reactants=_string_tuple(visual_raw.get("reactants")),
            reagents=_string_tuple(visual_raw.get("reagents")),
            conditions=_string_tuple(visual_raw.get("conditions")),
            stereochemistry=str(visual_raw.get("stereochemistry") or "").strip(),
            compounds=tuple(compounds),
            bounding_box=bounding_box,
            confidence=confidence,
            unreadable=bool(visual_raw.get("unreadable", False)),
            fallback_used=bool(visual_raw.get("fallback_used", False)),
        ),
    )


def manifest_from_dict(raw: dict[str, Any]) -> DocumentManifest:
    questions = [
        question for item in raw.get("questions") or []
        if isinstance(item, dict) and (question := question_from_dict(item)) is not None
    ]
    return DocumentManifest(
        schema_version=int(raw.get("schema_version") or 0),
        source_hash=str(raw.get("source_hash") or ""),
        model=str(raw.get("model") or ""),
        assignment_id=int(raw.get("assignment_id") or 0),
        suggested_topic=str(raw.get("suggested_topic") or "general"),
        questions=questions,
        created_at=str(raw.get("created_at") or ""),
        vision_request_count=int(raw.get("vision_request_count") or 0),
        fallback_crop_count=int(raw.get("fallback_crop_count") or 0),
        manifest_retry_count=int(raw.get("manifest_retry_count") or 0),
        retried_pages=tuple(
            int(value) for value in raw.get("retried_pages") or []
            if isinstance(value, int) and value > 0
        ),
    )


def load_manifest(path: Path) -> DocumentManifest:
    return manifest_from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_manifest(path: Path, manifest: DocumentManifest) -> None:
    path.write_text(json.dumps(manifest_to_dict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")


def with_cache_flag(manifest: DocumentManifest, from_cache: bool) -> DocumentManifest:
    return replace(manifest, from_cache=from_cache)


def manifest_text(manifest: DocumentManifest) -> str:
    lines = [f"Suggested topic: {manifest.suggested_topic}"]
    for question in manifest.questions:
        visual = question.visual
        lines.append(
            f"{question.id} {question.label}: {question.prompt_text or question.prompt_summary}; "
            f"visual={visual.description}; reactants={', '.join(visual.reactants)}; "
            f"reagents={', '.join(visual.reagents)}; conditions={', '.join(visual.conditions)}; "
            f"stereochemistry={visual.stereochemistry}"
        )
    return "\n".join(lines)
