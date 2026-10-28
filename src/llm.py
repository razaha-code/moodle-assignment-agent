import html
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .chemistry_render import compound_reference_html
from .pedigree import PEDIGREE_PROMPT, PEDIGREE_VERSION, parse_pedigrees, pedigree_html, requires_pedigree
from .llm_providers import LLMProviderRegistry
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    compute_source_hash,
    load_manifest,
    manifest_to_dict,
    question_from_dict,
    save_manifest,
    with_cache_flag,
)
from .models import (
    Assignment,
    AttachmentContext,
    CompoundReference,
    Config,
    DocumentManifest,
    GeneratedDocument,
    ManifestQuestion,
    QuestionAnswer,
    QuestionItem,
    UsageTotals,
    WorkflowError,
)
from .pdf_reader import crop_image_to_data_url


def strip_html(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value or "")
    unescaped = html.unescape(without_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def normalize_html_fragment(value: str) -> str:
    content = (value or "").strip()
    fenced_match = re.fullmatch(r"```(?:html)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        content = fenced_match.group(1).strip()

    body_match = re.search(r"<body[^>]*>(.*)</body>", content, flags=re.DOTALL | re.IGNORECASE)
    if body_match:
        content = body_match.group(1).strip()

    # Models sometimes wrap LaTeX delimiters in <code>. MathJax intentionally
    # ignores code elements, so unwrap only code nodes whose complete contents
    # are a math expression; ordinary code remains untouched.
    content = re.sub(
        r"<code\b[^>]*>\s*((?:\\\(.*?\\\)|\\\[.*?\\\]))\s*</code>",
        r"\1",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # \textsc is not part of the MathJax TeX configuration used by the output
    # document and renders as a red error. \mathrm preserves the intended LTR
    # language-name styling without requiring another MathJax extension.
    content = content.replace(r"\textsc{", r"\mathrm{")

    # A response can contain TeX commands inside an explicitly LTR span but
    # omit the opening math delimiter (for example ``A\in NP`` or
    # ``2\text{-}Coloring\)``). Those commands would otherwise be printed
    # literally. Wrap only plain-text LTR spans that clearly contain TeX;
    # spans with valid delimiters or nested markup are left unchanged.
    def wrap_bare_ltr_tex(match: re.Match[str]) -> str:
        opening, inner, closing = match.groups()
        if r"\(" in inner or r"\[" in inner:
            return match.group(0)
        stripped = inner.strip()
        has_tex_command = bool(re.search(r"\\[A-Za-z]+", stripped))
        has_orphan_closer = stripped.endswith(r"\)")
        if not (has_tex_command or has_orphan_closer):
            return match.group(0)
        if has_orphan_closer:
            stripped = stripped[:-2].rstrip()
        leading = inner[: len(inner) - len(inner.lstrip())]
        trailing = inner[len(inner.rstrip()) :]
        return f"{opening}{leading}\\({stripped}\\){trailing}{closing}"

    content = re.sub(
        r'(<span\b[^>]*\bdir=["\']ltr["\'][^>]*>)([^<]*)(</span>)',
        wrap_bare_ltr_tex,
        content,
        flags=re.IGNORECASE,
    )

    # Also repair an orphan ``\)`` inside a text node. This covers malformed
    # model HTML where the surrounding span itself was never closed, so the
    # span-based normalization above cannot safely match it.
    def repair_orphan_inline_closers(text_node: str) -> str:
        cursor = 0
        depth = 0
        repaired: list[str] = []
        while cursor < len(text_node):
            token = text_node[cursor : cursor + 2]
            if token == r"\(":
                depth += 1
                repaired.append(token)
                cursor += 2
                continue
            if token == r"\)":
                if depth:
                    depth -= 1
                else:
                    start = len(repaired)
                    while start and ord(repaired[start - 1][-1]) < 128:
                        start -= 1
                    while start < len(repaired) and repaired[start].isspace():
                        start += 1
                    if start < len(repaired):
                        repaired.insert(start, r"\(")
                repaired.append(token)
                cursor += 2
                continue
            repaired.append(text_node[cursor])
            cursor += 1
        return "".join(repaired)

    parts = re.split(r"(<[^>]+>)", content)
    content = "".join(
        part if part.startswith("<") else repair_orphan_inline_closers(part)
        for part in parts
    )

    return content.strip()


def is_substantive_answer_html(value: str) -> bool:
    text = strip_html(normalize_html_fragment(value)).strip()
    collapsed = re.sub(r"[\s.\u2026_\-–—]+", "", text).casefold()
    if not collapsed:
        return False
    placeholders = {
        "tbd", "todo", "placeholder", "answer", "na", "notavailable",
        "לאזמין", "טרםהושלם", "איןתשובה",
    }
    return collapsed not in placeholders


def is_acceptable_answer_html(value: str, answer_language: str = "auto") -> bool:
    if not is_substantive_answer_html(value):
        return False
    lowered = (value or "").casefold()
    leakage_markers = (
        "assistantanalysis", "assistant analysis", "weighted tokens left",
        "need final", "ensure json", "system message", "developer message",
    )
    if any(marker in lowered for marker in leakage_markers):
        return False
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", value or ""):
        return False
    if answer_language == "he":
        # Language validation applies to explanatory prose, not to formulas or
        # code. Counting identifiers such as x_i, c_i, 3-CNF and NAT as English
        # can otherwise reject a valid Hebrew proof with dense notation.
        prose_html = re.sub(
            r"<(code|pre)\b[^>]*>.*?</\1>",
            " ",
            value or "",
            flags=re.DOTALL | re.IGNORECASE,
        )
        prose_html = re.sub(
            r"\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\$[^$]*\$",
            " ",
            prose_html,
            flags=re.DOTALL,
        )
        text = strip_html(prose_html)
        hebrew_count = len(re.findall(r"[\u0590-\u05ff]", text))
        latin_count = len(re.findall(r"[A-Za-z]", text))
        return hebrew_count >= 20 and hebrew_count >= latin_count * 0.25
    return True


def normalize_manifest_identifier(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u0590-\u05ff]+", "", str(value or "").casefold())


def canonical_top_level(value: Any) -> str:
    text = str(value or "").strip()
    numeric = re.search(r"\d+", text)
    return numeric.group(0) if numeric else normalize_manifest_identifier(text)


def _manifest_response_format() -> dict[str, Any]:
    compound = {
        "type": "object",
        "properties": {
            "iupac_name": {"type": "string"},
            "smiles": {"type": "string"},
            "role": {"type": "string"},
        },
        "required": ["iupac_name", "smiles", "role"],
        "additionalProperties": False,
    }
    visual = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "reactants": {"type": "array", "items": {"type": "string"}},
            "reagents": {"type": "array", "items": {"type": "string"}},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "stereochemistry": {"type": "string"},
            "compounds": {"type": "array", "items": compound},
            "bounding_box": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    {"type": "null"},
                ]
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "unreadable": {"type": "boolean"},
        },
        "required": [
            "description", "reactants", "reagents", "conditions", "stereochemistry",
            "compounds", "bounding_box", "confidence", "unreadable",
        ],
        "additionalProperties": False,
    }
    question = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "top_level": {"type": "string"},
            "page_indices": {"type": "array", "items": {"type": "integer"}},
            "prompt_text": {"type": "string"},
            "prompt_summary": {"type": "string"},
            "visual": visual,
        },
        "required": [
            "id", "label", "top_level", "page_indices", "prompt_text", "prompt_summary", "visual",
        ],
        "additionalProperties": False,
    }
    audit = {
        "type": "object",
        "properties": {
            "page_index": {"type": "integer"},
            "fully_processed": {"type": "boolean"},
            "visible_top_levels": {"type": "array", "items": {"type": "string"}},
            "question_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["page_index", "fully_processed", "visible_top_levels", "question_ids"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "document_manifest",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "suggested_topic": {"type": "string"},
                    "page_audit": {"type": "array", "items": audit},
                    "questions": {"type": "array", "items": question},
                },
                "required": ["suggested_topic", "page_audit", "questions"],
                "additionalProperties": False,
            },
        },
    }


def _normalize_manifest_questions(raw_questions: Any) -> list[ManifestQuestion]:
    normalized: dict[str, ManifestQuestion] = {}
    for raw in raw_questions or []:
        if not isinstance(raw, dict):
            continue
        question = question_from_dict(raw)
        if question is None:
            continue
        item_id = normalize_manifest_identifier(question.id)
        top_level = canonical_top_level(question.top_level)
        if not item_id or not top_level or not question.page_indices:
            continue
        question.id = item_id
        question.top_level = top_level
        normalized[item_id] = question
    return list(normalized.values())


def _merge_manifest_question(
    existing: ManifestQuestion,
    incoming: ManifestQuestion,
) -> ManifestQuestion:
    """Merge duplicate normalized IDs without losing cross-page evidence."""
    existing.page_indices = tuple(sorted(set(existing.page_indices) | set(incoming.page_indices)))
    existing.top_level = incoming.top_level or existing.top_level
    existing.label = incoming.label or existing.label
    if len(incoming.prompt_text) > len(existing.prompt_text):
        existing.prompt_text = incoming.prompt_text
    if len(incoming.prompt_summary) > len(existing.prompt_summary):
        existing.prompt_summary = incoming.prompt_summary
    if incoming.visual.confidence > existing.visual.confidence or existing.visual.unreadable:
        existing.visual = incoming.visual
    return existing


def _validate_manifest_chunk(
    parsed: dict[str, Any],
    page_numbers: list[int],
    parsed_questions: list[ManifestQuestion],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"valid": True, "missing_pages": [], "pages": {}}
    if not page_numbers:
        return diagnostics
    audits = parsed.get("page_audit")
    if not isinstance(audits, list):
        diagnostics.update(valid=False, missing_pages=list(page_numbers), error="missing_page_audit")
        return diagnostics
    audits_by_page = {
        int(item.get("page_index")): item
        for item in audits
        if isinstance(item, dict) and isinstance(item.get("page_index"), int)
    }
    questions_by_id = {question.id: question for question in parsed_questions}
    represented_pages = {page for question in parsed_questions for page in question.page_indices}
    for page in page_numbers:
        audit = audits_by_page.get(page)
        reasons: list[str] = []
        if audit is None:
            reasons.append("audit_missing")
            diagnostics["pages"][str(page)] = {"valid": False, "reasons": reasons}
            diagnostics["missing_pages"].append(page)
            continue
        if audit.get("fully_processed") is not True:
            reasons.append("not_fully_processed")
        audited_ids = {
            normalize_manifest_identifier(value)
            for value in audit.get("question_ids") or []
            if normalize_manifest_identifier(value)
        }
        visible_top_levels = {
            normalize_manifest_identifier(value)
            for value in audit.get("visible_top_levels") or []
            if normalize_manifest_identifier(value)
        }
        page_records = [questions_by_id[item_id] for item_id in audited_ids if item_id in questions_by_id]
        unknown_ids = sorted(audited_ids - set(questions_by_id))
        if not audited_ids:
            reasons.append("question_ids_empty")
        if unknown_ids:
            reasons.append(f"unknown_question_ids:{','.join(unknown_ids)}")
        if page not in represented_pages:
            reasons.append("page_has_no_records")
        if not page_records or any(page not in question.page_indices for question in page_records):
            reasons.append("question_page_link_mismatch")
        record_top_levels = {question.top_level for question in page_records}
        if visible_top_levels and not visible_top_levels.issubset(record_top_levels):
            reasons.append("top_level_mismatch")
        page_valid = not reasons
        diagnostics["pages"][str(page)] = {
            "valid": page_valid,
            "reasons": reasons,
            "audited_question_ids": sorted(audited_ids),
            "record_question_ids": sorted(question.id for question in page_records),
        }
        if not page_valid:
            diagnostics["missing_pages"].append(page)
    diagnostics["valid"] = not diagnostics["missing_pages"]
    return diagnostics


def parse_json_object(value: str) -> dict[str, Any]:
    content = (value or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1).strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from the LLM.")
    return parsed


class LLMAnswerGenerator:
    def __init__(self, config: Config, provider_registry: LLMProviderRegistry | None = None) -> None:
        self.config = config
        self.provider_registry = provider_registry or LLMProviderRegistry(config)

    def usage_totals(self) -> UsageTotals:
        usage_method = getattr(self.provider_registry, "usage", None)
        return usage_method() if callable(usage_method) else UsageTotals()

    @staticmethod
    def _manifest_fingerprint(manifest: DocumentManifest) -> str:
        payload = manifest_to_dict(manifest)
        payload.pop("created_at", None)
        payload.pop("vision_request_count", None)
        payload.pop("fallback_crop_count", None)
        payload.pop("manifest_retry_count", None)
        payload.pop("retried_pages", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def load_or_build_manifest(
        self,
        assignment: Assignment,
        attachment_context: AttachmentContext,
        assignment_dir: Path,
        *,
        model: str | None = None,
        provider_name: str = "openai",
        rebuild: bool = False,
    ) -> tuple[DocumentManifest, Path]:
        resolved_model = (model or self.config.openai_model).strip()
        manifest_path = assignment_dir / "document_manifest.json"
        partial_path = assignment_dir / "document_manifest.partial.json"
        raw_manifest_path = assignment_dir / "document_manifest.raw.json"
        raw_retry_path = assignment_dir / "document_manifest_retry.raw.json"
        validation_path = assignment_dir / "manifest_validation.json"
        source_hash = compute_source_hash(attachment_context.saved_files, attachment_context.texts)
        if manifest_path.is_file() and not rebuild:
            try:
                cached = load_manifest(manifest_path)
                if (
                    cached.schema_version == MANIFEST_SCHEMA_VERSION
                    and cached.source_hash == source_hash
                    and cached.model == resolved_model
                    and cached.assignment_id == assignment.assignment_id
                    and cached.questions
                ):
                    return with_cache_flag(cached, True), manifest_path
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        provider = self.provider_registry.get(provider_name)
        source_text = "\n\n".join(value.strip() for value in attachment_context.texts if value.strip())[:12000]
        image_chunks = [
            (index, attachment_context.image_urls[index:index + 4])
            for index in range(0, len(attachment_context.image_urls), 4)
        ] or [(0, [])]
        questions: list[ManifestQuestion] = []
        seen_ids: set[str] = set()
        suggested_topic = "general"
        vision_requests = 0
        manifest_retry_count = 0
        retried_pages: set[int] = set()
        def load_raw_attempts(path: Path) -> list[dict[str, Any]]:
            if rebuild or not path.is_file():
                return []
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return [item for item in payload.get("attempts") or [] if isinstance(item, dict)]
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return []

        raw_attempts = load_raw_attempts(raw_manifest_path)
        retry_attempts = load_raw_attempts(raw_retry_path)
        raw_attempts = [
            item for item in raw_attempts if str(item.get("source_hash") or "") == source_hash
        ]
        retry_attempts = [
            item for item in retry_attempts if str(item.get("source_hash") or "") == source_hash
        ]
        validation_attempts: list[dict[str, Any]] = []
        completed_chunks: set[int] = set()
        if partial_path.is_file() and not rebuild:
            try:
                partial = json.loads(partial_path.read_text(encoding="utf-8"))
                if (
                    partial.get("schema_version") == MANIFEST_SCHEMA_VERSION
                    and partial.get("source_hash") == source_hash
                    and partial.get("model") == resolved_model
                    and partial.get("assignment_id") == assignment.assignment_id
                ):
                    completed_chunks = {int(value) for value in partial.get("completed_chunks") or []}
                    suggested_topic = str(partial.get("suggested_topic") or "general")
                    vision_requests = int(partial.get("vision_request_count") or 0)
                    manifest_retry_count = int(partial.get("manifest_retry_count") or 0)
                    retried_pages = {
                        int(value) for value in partial.get("retried_pages") or []
                        if isinstance(value, int) and value > 0
                    }
                    for raw in partial.get("questions") or []:
                        if isinstance(raw, dict):
                            question = question_from_dict(raw)
                            if question is not None and question.id not in seen_ids:
                                questions.append(question)
                                seen_ids.add(question.id)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                completed_chunks = set()
                questions = []
                seen_ids = set()

        # A validation-rule fix can make a previously rejected API response usable. Recover
        # complete chunks from the saved raw responses before sending any page image again.
        if not completed_chunks and raw_attempts and not rebuild:
            chunk_by_pages = {
                tuple(range(start_index + 1, start_index + len(images) + 1)): chunk_index
                for chunk_index, (start_index, images) in enumerate(image_chunks)
            }
            for raw_entry in raw_attempts:
                pages = tuple(
                    int(value) for value in raw_entry.get("pages") or []
                    if isinstance(value, int) and value > 0
                )
                chunk_index = chunk_by_pages.get(pages)
                parsed = raw_entry.get("response")
                if chunk_index is None or not isinstance(parsed, dict):
                    continue
                if str(raw_entry.get("model") or "") != resolved_model:
                    continue
                entry_hash = str(raw_entry.get("source_hash") or "")
                if entry_hash and entry_hash != source_hash:
                    continue
                chunk_questions = _normalize_manifest_questions(parsed.get("questions"))
                combined_parsed = dict(parsed)
                matched_retries = [
                    item for item in retry_attempts
                    if str(item.get("model") or "") == resolved_model
                    and set(item.get("pages") or []).issubset(set(pages))
                    and isinstance(item.get("response"), dict)
                    and (not item.get("source_hash") or item.get("source_hash") == source_hash)
                ]
                for retry_entry in matched_retries:
                    retry_parsed = retry_entry["response"]
                    retry_questions = _normalize_manifest_questions(retry_parsed.get("questions"))
                    retry_pages = {
                        int(value) for value in retry_entry.get("pages") or []
                        if isinstance(value, int) and value > 0
                    }
                    retry_ids = {question.id for question in retry_questions}
                    retained = [
                        question for question in chunk_questions
                        if not (
                            set(question.page_indices).issubset(retry_pages)
                            and question.id not in retry_ids
                        )
                    ]
                    merged = {question.id: question for question in retained}
                    for retry_question in retry_questions:
                        if retry_question.id in merged:
                            merged[retry_question.id] = _merge_manifest_question(
                                merged[retry_question.id], retry_question
                            )
                        else:
                            merged[retry_question.id] = retry_question
                    chunk_questions = list(merged.values())
                    audits_by_page = {
                        int(item["page_index"]): item
                        for item in combined_parsed.get("page_audit") or []
                        if isinstance(item, dict) and isinstance(item.get("page_index"), int)
                    }
                    for item in retry_parsed.get("page_audit") or []:
                        if isinstance(item, dict) and isinstance(item.get("page_index"), int):
                            audits_by_page[int(item["page_index"])] = item
                    combined_parsed["page_audit"] = list(audits_by_page.values())
                    retried_pages.update(retry_pages)

                recovered_validation = _validate_manifest_chunk(
                    combined_parsed, list(pages), chunk_questions
                )
                if not recovered_validation["valid"]:
                    continue
                validation_attempts.append({
                    "chunk_index": chunk_index,
                    "pages": list(pages),
                    "recovered_from_raw": True,
                    "final": recovered_validation,
                    "retried_pages": sorted(retried_pages.intersection(pages)),
                })
                suggested_topic = str(
                    combined_parsed.get("suggested_topic")
                    or parsed.get("suggested_topic")
                    or suggested_topic
                ).strip()
                for question in chunk_questions:
                    if question.id not in seen_ids:
                        questions.append(question)
                        seen_ids.add(question.id)
                    else:
                        existing = next(item for item in questions if item.id == question.id)
                        _merge_manifest_question(existing, question)
                completed_chunks.add(chunk_index)
                vision_requests += 1 + len(matched_retries)
                manifest_retry_count += len(matched_retries)

            if completed_chunks:
                validation_path.write_text(
                    json.dumps({"chunks": validation_attempts}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        def extract_pages(
            page_numbers: list[int],
            images: list[str],
            *,
            retry: bool,
        ) -> tuple[dict[str, Any], list[ManifestQuestion]]:
            retry_instruction = (
                "This is the single permitted retry. Extract only the attached pages and correct all omissions. "
                if retry else ""
            )
            prompt = (
                f"{retry_instruction}Digitize EVERY attached assignment page into one exhaustive, "
                "source-faithful manifest. Read pages in order and reset the top-level number whenever a new "
                "numbered question appears. Every atomic subpart or separate reaction must be a separate record. "
                "Do not solve. Transcribe all visible text and describe each diagram precisely. For chemistry "
                "record reactants, products shown in the source, reagents, conditions, stereochemistry, and only "
                "unambiguous IUPAC/SMILES. Each visual description must uniquely identify that atomic item. "
                "Never invent unreadable structures. bounding_box is normalized [x0,y0,x1,y1]. IDs combine the "
                "visible top-level number and part label; assign a, b, c in reading order to unlabeled reactions. "
                "Return one page_audit entry for every attached page and use the exact SOURCE PAGE number.\n"
                "Create inventory records only from the attached SOURCE PAGE images. Extracted text labeled "
                "REFERENCE SOLUTION is optional answer context and must not create additional question records.\n"
                f"Course: {assignment.course_name}\nAssignment: {assignment.title}\n"
                f"Pages in this request: {page_numbers or ['text-only']}\nExtracted text:\n{source_text}"
            )
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for page_number, url in zip(page_numbers, images):
                content.append({"type": "text", "text": f"SOURCE PAGE {page_number} (process fully):"})
                content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
            response = provider.generate(
                model=resolved_model,
                system_content="Create a lossless structured manifest of academic source pages.",
                user_content=content,
                temperature=0,
                reasoning_effort=self.config.openai_reasoning_effort,
                response_format=_manifest_response_format(),
                max_tokens=12000,
            )
            parsed = parse_json_object(response.content)
            raw_entry = {
                "pages": page_numbers,
                "model": resolved_model,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "source_hash": source_hash,
                "assignment_id": assignment.assignment_id,
                "request_id": response.request_id,
                "usage": response.usage.to_dict(),
                "response": parsed,
            }
            target_attempts = retry_attempts if retry else raw_attempts
            target_attempts.append(raw_entry)
            target_path = raw_retry_path if retry else raw_manifest_path
            target_path.write_text(
                json.dumps({"attempts": target_attempts}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return parsed, _normalize_manifest_questions(parsed.get("questions"))

        for chunk_index, (start_index, images) in enumerate(image_chunks):
            if chunk_index in completed_chunks:
                continue
            page_numbers = list(range(start_index + 1, start_index + len(images) + 1))
            parsed, chunk_questions = extract_pages(page_numbers, images, retry=False)
            if images:
                vision_requests += 1
            initial_validation = _validate_manifest_chunk(parsed, page_numbers, chunk_questions)
            final_validation = initial_validation
            missing_pages = list(initial_validation["missing_pages"])

            if missing_pages and images:
                retry_images = [images[page - start_index - 1] for page in missing_pages]
                retry_parsed, retry_questions = extract_pages(missing_pages, retry_images, retry=True)
                vision_requests += 1
                manifest_retry_count += 1
                retried_pages.update(missing_pages)
                provider_usage = getattr(provider, "usage_totals", None)
                if isinstance(provider_usage, UsageTotals):
                    provider_usage.manifest_retry_count += 1
                retry_ids = {question.id for question in retry_questions}
                retained = [
                    question for question in chunk_questions
                    if not (
                        set(question.page_indices).issubset(set(missing_pages))
                        and question.id not in retry_ids
                    )
                ]
                merged = {question.id: question for question in retained}
                for retry_question in retry_questions:
                    if retry_question.id in merged:
                        merged[retry_question.id] = _merge_manifest_question(
                            merged[retry_question.id], retry_question
                        )
                    else:
                        merged[retry_question.id] = retry_question
                chunk_questions = list(merged.values())
                audits_by_page = {
                    int(item["page_index"]): item
                    for item in parsed.get("page_audit") or []
                    if isinstance(item, dict) and isinstance(item.get("page_index"), int)
                }
                for item in retry_parsed.get("page_audit") or []:
                    if isinstance(item, dict) and isinstance(item.get("page_index"), int):
                        audits_by_page[int(item["page_index"])] = item
                combined_parsed = dict(parsed)
                combined_parsed["page_audit"] = list(audits_by_page.values())
                final_validation = _validate_manifest_chunk(
                    combined_parsed, page_numbers, chunk_questions
                )

            validation_attempts.append({
                "chunk_index": chunk_index,
                "pages": page_numbers,
                "initial": initial_validation,
                "final": final_validation,
                "retried_pages": missing_pages if missing_pages and images else [],
            })
            validation_path.write_text(
                json.dumps({"chunks": validation_attempts}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if not final_validation["valid"]:
                details = "; ".join(
                    f"page {page}: {', '.join(final_validation['pages'][str(page)]['reasons'])}"
                    for page in final_validation["missing_pages"]
                )
                failure = WorkflowError(f"Manifest validation failed after page retry: {details}")
                failure.manifest_validation_path = str(validation_path)
                failure.manifest_raw_path = str(raw_manifest_path)
                failure.manifest_retry_raw_path = str(raw_retry_path)
                failure.retried_pages = tuple(sorted(retried_pages))
                raise failure

            if suggested_topic == "general":
                suggested_topic = str(parsed.get("suggested_topic") or "general").strip()
            for question in chunk_questions:
                if question.id not in seen_ids:
                    questions.append(question)
                    seen_ids.add(question.id)
                else:
                    existing = next(item for item in questions if item.id == question.id)
                    _merge_manifest_question(existing, question)
            completed_chunks.add(chunk_index)
            partial_manifest = DocumentManifest(
                schema_version=MANIFEST_SCHEMA_VERSION,
                source_hash=source_hash,
                model=resolved_model,
                assignment_id=assignment.assignment_id,
                suggested_topic=suggested_topic or "general",
                questions=questions,
                created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                vision_request_count=vision_requests,
                manifest_retry_count=manifest_retry_count,
                retried_pages=tuple(sorted(retried_pages)),
            )
            partial_payload = manifest_to_dict(partial_manifest)
            partial_payload["completed_chunks"] = sorted(completed_chunks)
            partial_path.write_text(
                json.dumps(partial_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        if not questions:
            raise WorkflowError("The vision extraction pass did not identify any assignment questions.")
        manifest = DocumentManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            source_hash=source_hash,
            model=resolved_model,
            assignment_id=assignment.assignment_id,
            suggested_topic=suggested_topic or "general",
            questions=questions,
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            vision_request_count=vision_requests,
            fallback_crop_count=0,
            manifest_retry_count=manifest_retry_count,
            retried_pages=tuple(sorted(retried_pages)),
            from_cache=False,
        )
        save_manifest(manifest_path, manifest)
        if partial_path.is_file():
            partial_path.unlink()
        return manifest, manifest_path

    def refine_low_confidence_manifest_questions(
        self,
        manifest: DocumentManifest,
        manifest_path: Path,
        page_image_paths: list[str],
        *,
        provider_name: str,
        model: str | None,
    ) -> None:
        provider = self.provider_registry.get(provider_name)
        resolved_model = (model or self.config.openai_model).strip()
        for index, question in enumerate(list(manifest.questions)):
            visual = question.visual
            needs_fallback = visual.confidence < 0.8 or visual.unreadable
            if not needs_fallback or visual.fallback_used or visual.bounding_box is None:
                continue
            page_number = question.page_indices[0] if question.page_indices else 0
            if not 0 < page_number <= len(page_image_paths):
                continue
            crop_url = crop_image_to_data_url(Path(page_image_paths[page_number - 1]), visual.bounding_box)
            if not crop_url:
                continue
            prompt = (
                "Re-read this cropped question region and correct the manifest record. Do not solve it. "
                "Return JSON only as {\"question\":{...}} using the same question/visual schema. "
                f"Keep this stable ID: {question.id}. Existing transcription: {question.prompt_text}."
            )
            response = provider.generate(
                model=resolved_model,
                system_content="Correct one low-confidence manifest record from a tightly cropped source image.",
                user_content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": crop_url}},
                ],
                temperature=0,
                reasoning_effort=self.config.openai_reasoning_effort,
                response_format={"type": "json_object"},
                max_tokens=2500,
            )
            parsed = parse_json_object(response.content)
            raw_question = parsed.get("question")
            corrected = question_from_dict(raw_question, question.page_indices) if isinstance(raw_question, dict) else None
            if corrected is not None:
                corrected.id = question.id
                corrected.visual.fallback_used = True
                manifest.questions[index] = corrected
            else:
                question.visual.fallback_used = True
            manifest.fallback_crop_count += 1
            usage = getattr(provider, "usage_totals", None)
            if isinstance(usage, UsageTotals):
                usage.fallback_crop_count += 1
            save_manifest(manifest_path, manifest)

    def _generate_html_fragment(
        self,
        assignment: Assignment,
        attachment_texts: list[str],
        attachment_image_urls: list[str],
        task_request: str,
        system_content: str,
        provider_name: str = "openai",
        model: str | None = None,
        temperature: float = 0.2,
        reasoning_effort: str = "medium",
    ) -> str:
        intro_text = strip_html(assignment.intro_html).strip() or "No assignment description was provided."
        attachment_section = "\n\n".join(
            f"Attachment #{index + 1}:\n{text.strip()}"
            for index, text in enumerate(attachment_texts)
            if text.strip()
        )

        prompt = (
            "You are helping prepare a student-facing academic document.\n"
            f"{task_request}\n"
            "Return only an HTML fragment for the document body, not a full HTML page.\n"
            "Use semantic tags such as <section>, <h2>, <p>, <ol>, <ul>, <li>, <table>, <blockquote>, <pre>, and <code> when useful.\n"
            "For mathematics, preserve LaTeX exactly using inline \\(...\\) and display $$...$$ notation.\n"
            "For mixed Hebrew and English content, keep the natural wording and add dir=\"ltr\" only on clearly left-to-right snippets when needed.\n"
            "Do not mention system prompts, APIs, or hidden reasoning.\n\n"
            f"Course: {assignment.course_name}\n"
            f"Assignment: {assignment.title}\n"
            f"Due: {assignment.due_at.isoformat() if assignment.due_at else 'Unknown'}\n\n"
            f"Assignment description:\n{intro_text}\n\n"
            f"Attachment text:\n{attachment_section or 'No readable attachment text was available.'}\n"
            f"PDF page images attached: {len(attachment_image_urls)}\n"
        )

        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_url in attachment_image_urls:
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})

        resolved_model = (model or self.config.openai_model).strip()
        if not resolved_model:
            raise WorkflowError("The selected subject agent does not have an LLM model configured.")
        provider = self.provider_registry.get(provider_name)
        response = provider.generate(
            model=resolved_model,
            system_content=system_content,
            user_content=user_content,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        raw_content = response.content if hasattr(response, "content") else str(response)
        content = normalize_html_fragment(raw_content)
        if not content:
            raise WorkflowError(f"LLM provider {provider_name!r} returned an empty answer.")

        return content

    def generate_output_text(
        self,
        assignment: Assignment,
        attachment_texts: list[str],
        attachment_image_urls: list[str],
        mode: str,
    ) -> str:
        if mode == "instructions":
            task_request = (
                "Read the assignment carefully and write a structured instructions document for the student. "
                "Do not provide a final submission answer unless the PDF itself explicitly contains answers. "
                "Explain what the exercise asks, what information is given, recommended solving steps, important "
                "formulas or concepts, common traps, and what should be submitted."
            )
            system_content = (
                "Analyze assignment material and produce practical solving instructions as a clean HTML fragment. "
                "When PDF page images are provided, read them carefully and use them as the primary source."
            )
        else:
            task_request = (
                "Write a polished final answer document. If the assignment request is ambiguous or missing required "
                "source material, say so clearly and provide the best possible draft plus a short list of missing inputs."
            )
            system_content = (
                "Produce concise, correct assignment answers as a clean HTML fragment. "
                "When PDF page images are provided, read them carefully and use them as the primary source."
            )

        return self._generate_html_fragment(
            assignment,
            attachment_texts,
            attachment_image_urls,
            task_request,
            system_content,
        )

    def generate_specialist_output(
        self,
        assignment: Assignment,
        attachment_texts: list[str],
        attachment_image_urls: list[str],
        mode: str,
        specialist_name: str,
        specialist_instructions: str,
        provider_name: str = "openai",
        model: str | None = None,
        temperature: float = 0.2,
        reasoning_effort: str = "medium",
    ) -> str:
        if mode == "instructions":
            task_request = (
                f"Act as the {specialist_name} assigned to this document. "
                "Write a structured student-facing analysis and solving plan. Do not invent missing facts or "
                "provide a final submission answer. Identify every question, the relevant concepts, a concrete "
                "solution method, validation steps, common mistakes, and the expected submission format."
            )
        else:
            task_request = (
                f"Act as the {specialist_name} assigned to solve this document. "
                "Produce a complete question-by-question solution suitable for review before submission. Show "
                "the necessary reasoning and checks, preserve exact notation or code, and explicitly flag any "
                "part that cannot be read or solved from the supplied material."
            )

        system_content = (
            f"You are a specialized academic sub-agent: {specialist_name}.\n"
            f"Domain instructions:\n{specialist_instructions}\n"
            "Treat the supplied PDF page images as the primary source. Return only a clean HTML fragment."
        )
        return self._generate_html_fragment(
            assignment,
            attachment_texts,
            attachment_image_urls,
            task_request,
            system_content,
            provider_name,
            model,
            temperature,
            reasoning_effort,
        )

    def build_question_inventory(
        self,
        assignment: Assignment,
        attachment_texts: list[str],
        attachment_image_urls: list[str],
        *,
        provider_name: str,
        model: str | None,
        temperature: float,
        reasoning_effort: str = "medium",
    ) -> list[QuestionItem]:
        provider = self.provider_registry.get(provider_name)
        resolved_model = (model or self.config.openai_model).strip()
        source_text = "\n\n".join(text.strip() for text in attachment_texts if text.strip())[:12000]
        image_chunks = [
            (index, attachment_image_urls[index : index + 2])
            for index in range(0, len(attachment_image_urls), 2)
        ] or [(0, [])]
        inventory: list[QuestionItem] = []
        seen_ids: set[str] = set()

        for start_index, images in image_chunks:
            page_numbers = list(range(start_index + 1, start_index + len(images) + 1))
            prompt = (
                "Create an exhaustive atomic inventory of every question and subpart visible in these source "
                "pages. A question with parts (a)-(j) must produce ten separate items. Do not solve anything. "
                "Return JSON only: {\"questions\":[{\"id\":\"2a\",\"label\":\"(a)\","
                "\"top_level\":\"2\",\"page_indices\":[1],\"prompt_summary\":\"...\"}]}. "
                "Use stable IDs, preserve source order, and include Hebrew-labeled subparts too.\n"
                f"Course: {assignment.course_name}\nAssignment: {assignment.title}\n"
                f"Pages in this request: {page_numbers or ['text-only']}\nExtracted text:\n{source_text}"
            )
            user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            user_content.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
            parsed = None
            for _ in range(2):
                response = provider.generate(
                    model=resolved_model,
                    system_content="You extract complete question inventories from academic assignments.",
                    user_content=user_content,
                    temperature=0,
                    reasoning_effort=reasoning_effort,
                    response_format={"type": "json_object"},
                    max_tokens=2500,
                )
                try:
                    parsed = parse_json_object(response.content)
                    break
                except (json.JSONDecodeError, ValueError):
                    continue
            if parsed is None:
                raise WorkflowError(f"Could not build a question inventory for pages {page_numbers}.")

            for raw in parsed.get("questions", []):
                if not isinstance(raw, dict):
                    continue
                item_id = str(raw.get("id", "")).strip()
                if not item_id or item_id in seen_ids:
                    continue
                raw_pages = raw.get("page_indices")
                pages = tuple(
                    int(page) for page in raw_pages
                    if isinstance(page, int) and page > 0
                ) if isinstance(raw_pages, list) else tuple(page_numbers)
                inventory.append(
                    QuestionItem(
                        id=item_id,
                        label=str(raw.get("label") or item_id).strip(),
                        top_level=str(raw.get("top_level") or item_id).strip(),
                        page_indices=pages or tuple(page_numbers),
                        prompt_summary=str(raw.get("prompt_summary") or "").strip(),
                    )
                )
                seen_ids.add(item_id)

        if not inventory:
            raise WorkflowError("The LLM did not identify any questions in the assignment source.")
        return inventory

    def _review_pedigrees(self, pedigrees, source, provider, resolved_model, reasoning_effort, answer_html=""):
        if not pedigrees:
            return []
        review = provider.generate(
            model=resolved_model,
            system_content="Audit pedigree facts against the original assignment. Do not solve unrelated items.",
            user_content=[{"type": "text", "text":
                "Return JSON {\"errors\":[\"specific mismatch\"]}. Check completeness, sex, disease versus "
                "carrier status, exact parents and partners, and invented individuals. Check every evidence "
                "quote actually supports its fact, not merely that it occurs in the source. Brother means male. "
                "Reject omitted relatives or hypothetical children. Check the accompanying explanation for "
                "contradictions with the pedigree/source and reject any duplicate ASCII family diagram. "
                "Empty errors means all facts agree.\n"
                + source + "\n" + json.dumps([p.to_dict() for p in pedigrees], ensure_ascii=False)
                + "\nAccompanying explanation:\n" + answer_html}],
            temperature=0, reasoning_effort=reasoning_effort,
            response_format={"type": "json_object"}, max_tokens=2000,
        )
        try:
            errors = parse_json_object(review.content)["errors"]
            if review.finish_reason == "length" or not isinstance(errors, list) or any(not isinstance(e, str) for e in errors):
                raise ValueError("invalid review")
            return errors
        except (ValueError, KeyError, TypeError):
            return ["Pedigree source verification did not return a complete valid review"]

    def generate_specialist_document(
        self,
        assignment: Assignment,
        attachment_texts: list[str],
        attachment_image_urls: list[str],
        mode: str,
        specialist_name: str,
        specialist_instructions: str,
        *,
        provider_name: str = "openai",
        model: str | None = None,
        temperature: float = 0.2,
        reasoning_effort: str = "medium",
        answer_language: str = "auto",
    ) -> GeneratedDocument:
        inventory = self.build_question_inventory(
            assignment,
            attachment_texts,
            attachment_image_urls,
            provider_name=provider_name,
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        provider = self.provider_registry.get(provider_name)
        resolved_model = (model or self.config.openai_model).strip()
        source_text = "\n\n".join(text.strip() for text in attachment_texts if text.strip())[:12000]
        expected_by_id = {item.id: item for item in inventory}
        answers_by_id: dict[str, QuestionAnswer] = {}
        attempts = {item.id: 0 for item in inventory}
        finish_reasons: list[str] = []
        pedigree_errors: dict[str, list[str]] = {}

        grouped: list[list[QuestionItem]] = []
        for item in inventory:
            if not grouped or grouped[-1][0].top_level != item.top_level or len(grouped[-1]) >= 4:
                grouped.append([item])
            else:
                grouped[-1].append(item)

        task = "write solving instructions" if mode == "instructions" else "solve completely"

        def solve_batch(items: list[QuestionItem]) -> None:
            for item in items:
                attempts[item.id] += 1
            requested_ids = [item.id for item in items]
            item_descriptions = "\n".join(
                f"- {item.id} ({item.label}): {item.prompt_summary}" for item in items
            )
            relevant_pages = sorted({page for item in items for page in item.page_indices})
            selected_images = [
                attachment_image_urls[page - 1]
                for page in relevant_pages
                if 0 < page <= len(attachment_image_urls)
            ]
            prompt = (
                f"You are the {specialist_name}. {task} only the requested atomic items below. "
                "Return every requested ID exactly once. Return JSON only with this schema: "
                "{\"answers\":[{\"id\":\"2a\",\"html\":\"<p>...</p>\","
                "\"compounds\":[{\"iupac_name\":\"English IUPAC name\","
                "\"smiles\":\"SMILES\",\"role\":\"product\"}]}]}. "
                "The compounds list is optional and should be empty outside relevant molecular questions. "
                "Do not invent unreadable structures. Answer prose may be Hebrew when required by the domain "
                "instructions, but IUPAC names and SMILES must remain English left-to-right.\n"
                f"{PEDIGREE_PROMPT}\nPrior pedigree errors: {json.dumps(pedigree_errors, ensure_ascii=False)}\n"
                f"Domain instructions:\n{specialist_instructions}\nRequested items:\n{item_descriptions}\n"
                f"Extracted source text:\n{source_text}"
            )
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            content.extend({"type": "image_url", "image_url": {"url": url}} for url in selected_images)
            response = provider.generate(
                model=resolved_model,
                system_content="Produce complete, source-grounded academic answers as structured JSON.",
                user_content=content,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                response_format={"type": "json_object"},
                max_tokens=8000,
            )
            finish_reasons.append(response.finish_reason)
            if response.finish_reason == "length" and len(items) > 1:
                midpoint = len(items) // 2
                solve_batch(items[:midpoint])
                solve_batch(items[midpoint:])
                return
            try:
                parsed = parse_json_object(response.content)
            except (json.JSONDecodeError, ValueError):
                if len(items) > 1:
                    midpoint = len(items) // 2
                    solve_batch(items[:midpoint])
                    solve_batch(items[midpoint:])
                return
            for raw in parsed.get("answers", []):
                if not isinstance(raw, dict):
                    continue
                item_id = str(raw.get("id", "")).strip()
                answer_html = normalize_html_fragment(str(raw.get("html", "")))
                if item_id not in requested_ids or not is_acceptable_answer_html(
                    answer_html, answer_language
                ):
                    continue
                compounds: list[CompoundReference] = []
                for compound in raw.get("compounds") or []:
                    if not isinstance(compound, dict):
                        continue
                    iupac = str(compound.get("iupac_name") or "").strip()
                    smiles = str(compound.get("smiles") or "").strip()
                    if iupac or smiles:
                        compounds.append(
                            CompoundReference(
                                iupac_name=iupac,
                                smiles=smiles,
                                role=str(compound.get("role") or "").strip(),
                            )
                        )
                try:
                    pedigrees = parse_pedigrees(raw.get("pedigrees"), source_text)
                    if mode == "solve" and requires_pedigree(expected_by_id[item_id].prompt_summary) and not pedigrees:
                        raise ValueError("Structured pedigree required")
                    errors = self._review_pedigrees(pedigrees, source_text, provider, resolved_model, reasoning_effort, answer_html)
                    if errors:
                        pedigree_errors[item_id] = errors
                        continue
                except (ValueError, TypeError) as exc:
                    pedigree_errors[item_id] = [str(exc)]
                    continue
                pedigree_errors.pop(item_id, None)
                answers_by_id[item_id] = QuestionAnswer(item_id, answer_html, tuple(compounds), pedigrees)

        for batch in grouped:
            solve_batch(batch)

        for _ in range(2):
            missing = [
                item for item in inventory
                if item.id not in answers_by_id and attempts[item.id] < 3
            ]
            if not missing:
                break
            for item in missing:
                solve_batch([item])

        ordered_answers = [answers_by_id[item.id] for item in inventory if item.id in answers_by_id]
        missing_ids = [item.id for item in inventory if item.id not in answers_by_id]
        sections: list[str] = ['<section class="assignment-answers">']
        current_top_level = None
        for item in inventory:
            if item.top_level != current_top_level:
                sections.append(f"<h2>שאלה {html.escape(canonical_top_level(item.top_level))}</h2>")
                current_top_level = item.top_level
            sections.append(
                f'<article class="question-answer" data-question-id="{html.escape(item.id)}">'
                f"<h3>{html.escape(item.label)}</h3>"
            )
            answer = answers_by_id.get(item.id)
            if answer is None:
                sections.append('<p class="missing-answer">הסעיף לא הושלם לאחר ניסיונות חוזרים.</p>')
            else:
                sections.append(answer.html)
                sections.extend(pedigree_html(p) for p in answer.pedigrees)
                if answer.compounds:
                    sections.append('<aside class="compound-references" dir="ltr"><h4>Compounds</h4><ul>')
                    for compound in answer.compounds:
                        label = compound.iupac_name or "Unnamed compound"
                        details = f" - SMILES: {compound.smiles}" if compound.smiles else ""
                        role = f" ({compound.role})" if compound.role else ""
                        sections.append(f"<li>{html.escape(label + role + details)}</li>")
                    sections.append("</ul></aside>")
            sections.append("</article>")
        sections.append("</section>")
        return GeneratedDocument(
            html="".join(sections),
            inventory=inventory,
            answers=ordered_answers,
            missing_ids=missing_ids,
            attempts=attempts,
            finish_reasons=finish_reasons,
        )

    def generate_specialist_document_from_manifest(
        self,
        assignment: Assignment,
        manifest: DocumentManifest,
        mode: str,
        specialist_name: str,
        specialist_instructions: str,
        *,
        checkpoint_path: Path,
        manifest_path: Path,
        page_image_paths: list[str],
        provider_name: str = "openai",
        model: str | None = None,
        temperature: float = 0.2,
        reasoning_effort: str = "medium",
        answer_language: str = "auto",
    ) -> GeneratedDocument:
        self.refine_low_confidence_manifest_questions(
            manifest,
            manifest_path,
            page_image_paths,
            provider_name=provider_name,
            model=model,
        )
        provider = self.provider_registry.get(provider_name)
        resolved_model = (model or self.config.openai_model).strip()
        inventory = [question.to_question_item() for question in manifest.questions]
        manifest_by_id = {question.id: question for question in manifest.questions}
        answers_by_id: dict[str, QuestionAnswer] = {}
        attempts = {item.id: 0 for item in inventory}
        finish_reasons: list[str] = []
        resumed = False
        manifest_fingerprint = self._manifest_fingerprint(manifest)
        pedigree_errors: dict[str, list[str]] = {}
        pedigree_sources = {
            q.id: "\n".join(s.prompt_text for s in manifest.questions if s.top_level == q.top_level)
            for q in manifest.questions
        }

        def read_pedigrees(raw, item_id):
            pedigrees = parse_pedigrees(raw.get("pedigrees"), pedigree_sources[item_id])
            if mode == "solve" and requires_pedigree(manifest_by_id[item_id].prompt_text) and not pedigrees:
                raise ValueError("This item requires structured pedigrees; ASCII trees are not accepted")
            return pedigrees


        if checkpoint_path.is_file():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if (
                    checkpoint.get("source_hash") == manifest.source_hash
                    and checkpoint.get("manifest_fingerprint") == manifest_fingerprint
                    and checkpoint.get("model") == resolved_model
                    and checkpoint.get("mode") == mode
                    and checkpoint.get("specialist_name") == specialist_name
                ):
                    for item_id, count in (checkpoint.get("attempts") or {}).items():
                        if item_id in attempts:
                            attempts[item_id] = int(count)
                    finish_reasons.extend(str(value) for value in checkpoint.get("finish_reasons") or [])
                    for raw in checkpoint.get("answers") or []:
                        if not isinstance(raw, dict):
                            continue
                        item_id = str(raw.get("id") or "")
                        if item_id not in attempts:
                            continue
                        compounds = tuple(
                            CompoundReference(
                                iupac_name=str(compound.get("iupac_name") or ""),
                                smiles=str(compound.get("smiles") or ""),
                                role=str(compound.get("role") or ""),
                            )
                            for compound in raw.get("compounds") or []
                            if isinstance(compound, dict)
                        )
                        answer_html = normalize_html_fragment(str(raw.get("html") or ""))
                        try:
                            pedigrees = read_pedigrees(raw, item_id)
                            if pedigrees and checkpoint.get("pedigree_version") != PEDIGREE_VERSION:
                                raise ValueError("cached pedigree requires source verification")
                        except (ValueError, TypeError) as exc:
                            pedigree_errors[item_id] = [str(exc)]
                            if checkpoint.get("pedigree_version") != PEDIGREE_VERSION:
                                attempts[item_id] = 0
                            continue
                        if is_acceptable_answer_html(answer_html, answer_language):
                            answers_by_id[item_id] = QuestionAnswer(item_id, answer_html, compounds, pedigrees)
                    resumed = bool(answers_by_id)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        def save_checkpoint() -> None:
            payload = {
                "schema_version": 2,
                "pedigree_version": PEDIGREE_VERSION,
                "pedigree_errors": pedigree_errors,
                "source_hash": manifest.source_hash,
                "manifest_fingerprint": manifest_fingerprint,
                "model": resolved_model,
                "mode": mode,
                "specialist_name": specialist_name,
                "attempts": attempts,
                "finish_reasons": finish_reasons,
                "answers": [
                    {
                        "id": answer.id,
                        "html": answer.html,
                        "pedigrees": [p.to_dict() for p in answer.pedigrees],
                        "compounds": [
                            {"iupac_name": value.iupac_name, "smiles": value.smiles, "role": value.role}
                            for value in answer.compounds
                        ],
                    }
                    for answer in answers_by_id.values()
                ],
            }
            checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        grouped: list[list[QuestionItem]] = []
        pending_inventory = [item for item in inventory if item.id not in answers_by_id]
        for item in pending_inventory:
            if not grouped or grouped[-1][0].top_level != item.top_level or len(grouped[-1]) >= 4:
                grouped.append([item])
            else:
                grouped[-1].append(item)

        task = "write solving instructions" if mode == "instructions" else "solve completely"
        language_instruction = (
            "Write all prose, proof steps, explanations, and conclusions in Hebrew. "
            "Keep only standard technical terms, symbols, and formulas in English/LTR. "
            if answer_language == "he" else ""
        )

        def solve_batch(items: list[QuestionItem]) -> None:
            active_items = [item for item in items if item.id not in answers_by_id and attempts[item.id] < 3]
            if not active_items:
                return
            for item in active_items:
                attempts[item.id] += 1
            requested_ids = [item.id for item in active_items]
            active_top_levels = {item.top_level for item in active_items}
            active_pages = {
                page
                for item in active_items
                for page in manifest_by_id[item.id].page_indices
            }
            # A requested sub-question can refer to a definition, construction or
            # diagram stated in a sibling/adjacent record. Include records from
            # the same top-level question or source page as read-only context,
            # while still asking the model to answer only the active IDs.
            context_items = [
                item
                for item in inventory
                if item.top_level in active_top_levels
                or active_pages.intersection(manifest_by_id[item.id].page_indices)
            ]
            records = []
            for item in context_items:
                question = manifest_by_id[item.id]
                records.append(
                    {
                        "id": question.id,
                        "answer_requested": question.id in requested_ids,
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
                            "compounds": [
                                {"iupac_name": value.iupac_name, "smiles": value.smiles, "role": value.role}
                                for value in question.visual.compounds
                            ],
                            "confidence": question.visual.confidence,
                            "unreadable": question.visual.unreadable,
                        },
                    }
                )
            prompt = (
                f"You are the {specialist_name}. {task} only the requested atomic items. "
                "The source images were digitized once into the records below; do not request or assume other "
                "visual information. Records with answer_requested=false are context only. Return every requested "
                "ID exactly once and do not answer context-only IDs. Return JSON: "
                "{\"answers\":[{\"id\":\"1a\",\"html\":\"<p>Complete explained answer</p>\",\"compounds\":["
                "{\"iupac_name\":\"English IUPAC name\",\"smiles\":\"SMILES\",\"role\":\"product\"}]}]}. "
                "Never return ellipses, placeholders, or a bare answer without reasoning. For chemistry include "
                "only the actual organic reactant/product structures in compounds; never list a reagent, base, "
                "salt, or solvent as the product. Use valid isomeric SMILES. Flag unreadable source facts instead "
                "of inventing them. IUPAC and SMILES remain English LTR.\n"
                f"{PEDIGREE_PROMPT}\nPrior pedigree errors to correct: {json.dumps(pedigree_errors, ensure_ascii=False)}\n"
                f"{language_instruction}Domain instructions:\n{specialist_instructions}\nManifest records:\n"
                f"{json.dumps(records, ensure_ascii=False)}"
            )
            response = provider.generate(
                model=resolved_model,
                system_content="Produce complete source-grounded academic answers from a verified manifest.",
                user_content=[{"type": "text", "text": prompt}],
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                response_format={"type": "json_object"},
                max_tokens=8000,
            )
            finish_reasons.append(response.finish_reason)
            if response.finish_reason == "length" and len(active_items) > 1:
                save_checkpoint()
                midpoint = len(active_items) // 2
                solve_batch(active_items[:midpoint])
                solve_batch(active_items[midpoint:])
                return
            try:
                parsed = parse_json_object(response.content)
            except (json.JSONDecodeError, ValueError):
                save_checkpoint()
                if len(active_items) > 1:
                    midpoint = len(active_items) // 2
                    solve_batch(active_items[:midpoint])
                    solve_batch(active_items[midpoint:])
                return
            for raw in parsed.get("answers") or []:
                if not isinstance(raw, dict):
                    continue
                item_id = normalize_manifest_identifier(raw.get("id"))
                answer_html = normalize_html_fragment(str(raw.get("html") or ""))
                if item_id not in requested_ids or not is_acceptable_answer_html(
                    answer_html, answer_language
                ):
                    continue
                compounds = tuple(
                    CompoundReference(
                        iupac_name=str(value.get("iupac_name") or "").strip(),
                        smiles=str(value.get("smiles") or "").strip(),
                        role=str(value.get("role") or "").strip(),
                    )
                    for value in raw.get("compounds") or []
                    if isinstance(value, dict)
                )
                try:
                    pedigrees = read_pedigrees(raw, item_id)
                    errors = self._review_pedigrees(pedigrees, pedigree_sources[item_id], provider, resolved_model, reasoning_effort, answer_html)
                    if errors:
                        pedigree_errors[item_id] = errors
                        continue
                except (ValueError, TypeError) as exc:
                    pedigree_errors[item_id] = [str(exc)]
                    continue
                pedigree_errors.pop(item_id, None)
                answers_by_id[item_id] = QuestionAnswer(item_id, answer_html, compounds, pedigrees)
            save_checkpoint()

        for batch in grouped:
            solve_batch(batch)
        for _ in range(2):
            missing = [
                item for item in inventory
                if item.id not in answers_by_id and attempts[item.id] < 3
            ]
            if not missing:
                break
            for item in missing:
                solve_batch([item])

        ordered_answers = [answers_by_id[item.id] for item in inventory if item.id in answers_by_id]
        missing_ids = [item.id for item in inventory if item.id not in answers_by_id]
        sections: list[str] = ['<section class="assignment-answers">']
        current_top_level = None
        for item in inventory:
            if item.top_level != current_top_level:
                sections.append(f"<h2>שאלה {html.escape(canonical_top_level(item.top_level))}</h2>")
                current_top_level = item.top_level
            sections.append(
                f'<article class="question-answer" data-question-id="{html.escape(item.id)}">'
                f"<h3>{html.escape(item.label)}</h3>"
            )
            answer = answers_by_id.get(item.id)
            if answer is None:
                sections.append('<p class="missing-answer">הסעיף לא הושלם לאחר ניסיונות חוזרים.</p>')
            else:
                sections.append(answer.html)
                sections.extend(pedigree_html(p) for p in answer.pedigrees)
                if answer.compounds:
                    sections.append('<aside class="compound-references" dir="ltr"><h4>Chemical structures</h4>')
                    for compound in answer.compounds:
                        sections.append(compound_reference_html(
                            compound.iupac_name, compound.smiles, compound.role
                        ))
                    sections.append("</aside>")
            sections.append("</article>")
        sections.append("</section>")
        save_checkpoint()
        return GeneratedDocument(
            html="".join(sections),
            inventory=inventory,
            answers=ordered_answers,
            missing_ids=missing_ids,
            attempts=attempts,
            finish_reasons=finish_reasons,
            resumed_from_checkpoint=resumed,
            manifest_path=str(manifest_path),
            checkpoint_path=str(checkpoint_path),
        )

    def generate_solution_from_instructions(
        self,
        assignment: Assignment,
        instructions_text: str,
        attachment_image_urls: list[str],
        specialist_name: str = "general academic solver",
        specialist_instructions: str = "Solve carefully and verify the result against the source document.",
        provider_name: str = "openai",
        model: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        task_request = (
            "You are the solver agent in a two-stage workflow. Stage 1 already analyzed the assignment PDF and "
            "produced instructions. Produce a concrete, well-structured solution draft based on those instructions "
            "and the attached PDF page images. If a question is unreadable or ambiguous, say exactly which part is unclear."
        )
        system_content = (
            f"You are the {specialist_name} solver sub-agent. {specialist_instructions} "
            "Read the instructions and assignment PDF images carefully, then write a question-by-question "
            "solution as a clean HTML fragment with LaTeX preserved for mathematics."
        )

        return self._generate_html_fragment(
            assignment,
            [instructions_text],
            attachment_image_urls,
            task_request,
            system_content,
            provider_name,
            model,
            temperature,
        )


# Compatibility alias for imports from earlier versions of the project.
OpenAIAnswerGenerator = LLMAnswerGenerator
