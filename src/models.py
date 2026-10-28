import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .pedigree import Pedigree


class WorkflowError(RuntimeError):
    pass


class WorkflowStageError(WorkflowError):
    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"{stage}: {cause}")
        self.stage = stage
        self.cause = cause


@dataclass
class Config:
    moodle_base_url: str
    moodle_token: str
    moodle_service: str
    openai_api_key: str
    openai_model: str
    timezone: str
    active_course_ids: set[int]
    openai_reasoning_effort: str = "medium"


@dataclass
class SelectionWindow:
    mode: str
    start_date: dt.date
    end_date: dt.date


@dataclass
class AttachmentContext:
    texts: list[str]
    image_urls: list[str]
    saved_files: list[dict[str, Any]]
    preview_image_paths: list[str]


@dataclass
class RuntimeDiagnostics:
    env_file_path: str
    env_file_exists: bool
    openai_api_key_present: bool
    moodle_token_present: bool
    moodle_username_present: bool
    moodle_password_present: bool
    active_course_ids_present: bool
    reportlab_available: bool
    reportlab_font_path: str | None
    playwright_available: bool
    output_dir: str
    token_bootstrap_available: bool
    pymupdf_available: bool


@dataclass
class Assignment:
    course_id: int
    course_name: str
    assignment_id: int
    title: str
    intro_html: str
    due_at: dt.datetime | None
    allow_submissions_from: dt.datetime | None
    cutoff_at: dt.datetime | None
    attachments: list[dict[str, Any]]
    already_submitted: bool = False
    submission_status: str | None = None
    submission_last_modified: dt.datetime | None = None


@dataclass(frozen=True)
class QuestionItem:
    id: str
    label: str
    top_level: str
    page_indices: tuple[int, ...]
    prompt_summary: str


@dataclass
class UsageTotals:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    request_count: int = 0
    vision_request_count: int = 0
    fallback_crop_count: int = 0
    manifest_retry_count: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: "UsageTotals") -> None:
        self.input_tokens += other.input_tokens
        self.cached_tokens += other.cached_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.request_count += other.request_count
        self.vision_request_count += other.vision_request_count
        self.fallback_crop_count += other.fallback_crop_count
        self.manifest_retry_count += other.manifest_retry_count
        self.estimated_cost_usd += other.estimated_cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "vision_request_count": self.vision_request_count,
            "fallback_crop_count": self.fallback_crop_count,
            "manifest_retry_count": self.manifest_retry_count,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


@dataclass
class VisualEvidence:
    description: str = ""
    reactants: tuple[str, ...] = field(default_factory=tuple)
    reagents: tuple[str, ...] = field(default_factory=tuple)
    conditions: tuple[str, ...] = field(default_factory=tuple)
    stereochemistry: str = ""
    compounds: tuple["CompoundReference", ...] = field(default_factory=tuple)
    bounding_box: tuple[float, float, float, float] | None = None
    confidence: float = 1.0
    unreadable: bool = False
    fallback_used: bool = False


@dataclass
class ManifestQuestion:
    id: str
    label: str
    top_level: str
    page_indices: tuple[int, ...]
    prompt_text: str
    prompt_summary: str
    visual: VisualEvidence = field(default_factory=VisualEvidence)

    def to_question_item(self) -> QuestionItem:
        return QuestionItem(
            id=self.id,
            label=self.label,
            top_level=self.top_level,
            page_indices=self.page_indices,
            prompt_summary=self.prompt_summary or self.prompt_text,
        )


@dataclass
class DocumentManifest:
    schema_version: int
    source_hash: str
    model: str
    assignment_id: int
    suggested_topic: str
    questions: list[ManifestQuestion]
    created_at: str
    vision_request_count: int = 0
    fallback_crop_count: int = 0
    manifest_retry_count: int = 0
    retried_pages: tuple[int, ...] = field(default_factory=tuple)
    from_cache: bool = False


@dataclass(frozen=True)
class CompoundReference:
    iupac_name: str
    smiles: str
    role: str = ""


@dataclass(frozen=True)
class QuestionAnswer:
    id: str
    html: str
    compounds: tuple[CompoundReference, ...] = field(default_factory=tuple)
    pedigrees: tuple[Pedigree, ...] = field(default_factory=tuple)


@dataclass
class GeneratedDocument:
    html: str
    inventory: list[QuestionItem]
    answers: list[QuestionAnswer]
    missing_ids: list[str]
    attempts: dict[str, int]
    finish_reasons: list[str]
    resumed_from_checkpoint: bool = False
    manifest_path: str | None = None
    checkpoint_path: str | None = None

    @property
    def complete(self) -> bool:
        return not self.missing_ids

    def coverage_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "expected_count": len(self.inventory),
            "answered_count": len(self.answers),
            "missing_ids": self.missing_ids,
            "attempts": self.attempts,
            "finish_reasons": self.finish_reasons,
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "manifest_path": self.manifest_path,
            "checkpoint_path": self.checkpoint_path,
            "inventory": [
                {
                    "id": item.id,
                    "label": item.label,
                    "top_level": item.top_level,
                    "page_indices": list(item.page_indices),
                    "prompt_summary": item.prompt_summary,
                }
                for item in self.inventory
            ],
            "answers": [
                {
                    "id": answer.id,
                    "pedigrees": [p.to_dict() for p in answer.pedigrees],
                    "compounds": [
                        {
                            "iupac_name": compound.iupac_name,
                            "smiles": compound.smiles,
                            "role": compound.role,
                        }
                        for compound in answer.compounds
                    ],
                }
                for answer in self.answers
            ],
        }
