from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .llm import LLMAnswerGenerator
from .models import Assignment, DocumentManifest, GeneratedDocument

if TYPE_CHECKING:
    from .agent_loader import AgentRegistry


@dataclass(frozen=True)
class SubjectAgent:
    topic: str
    display_name: str
    instructions: str
    keywords: tuple[str, ...] = field(default_factory=tuple)
    aliases: tuple[str, ...] = field(default_factory=tuple)
    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.2
    reasoning_effort: str = "medium"
    answer_language: str = "auto"
    output_style: str = "standard"
    paper_style: str = "plain"

    def generate(
        self,
        generator: LLMAnswerGenerator,
        assignment: Assignment,
        attachment_texts: list[str],
        attachment_image_urls: list[str],
        mode: str,
    ) -> GeneratedDocument:
        return generator.generate_specialist_document(
            assignment,
            attachment_texts,
            attachment_image_urls,
            mode,
            self.display_name,
            self.instructions,
            provider_name=self.provider,
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            answer_language=self.answer_language,
        )

    def solve_from_instructions(
        self,
        generator: LLMAnswerGenerator,
        assignment: Assignment,
        instructions_text: str,
        attachment_image_urls: list[str],
    ) -> GeneratedDocument:
        return generator.generate_specialist_document(
            assignment,
            [instructions_text],
            attachment_image_urls,
            "solve",
            self.display_name,
            self.instructions,
            provider_name=self.provider,
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            answer_language=self.answer_language,
        )

    def generate_from_manifest(
        self,
        generator: LLMAnswerGenerator,
        assignment: Assignment,
        manifest: DocumentManifest,
        mode: str,
        *,
        checkpoint_path: Path,
        manifest_path: Path,
        page_image_paths: list[str],
    ) -> GeneratedDocument:
        return generator.generate_specialist_document_from_manifest(
            assignment,
            manifest,
            mode,
            self.display_name,
            self.instructions,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            page_image_paths=page_image_paths,
            provider_name=self.provider,
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            answer_language=self.answer_language,
        )


def get_subject_agent(topic: str, registry: "AgentRegistry | None" = None) -> SubjectAgent:
    if registry is None:
        from .agent_loader import get_agent_registry

        registry = get_agent_registry()
    return registry.get(topic)
