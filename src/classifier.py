import json
from dataclasses import dataclass
from typing import Any

import requests

from .agent_loader import AgentRegistry, get_agent_registry
from .llm_providers import OpenAIProvider
from .models import Assignment, Config, UsageTotals, WorkflowError


@dataclass(frozen=True)
class TopicClassification:
    topic: str
    confidence: float
    reason: str
    method: str


def normalize_topic(value: str, registry: AgentRegistry | None = None) -> str:
    return (registry or get_agent_registry()).normalize_topic(value)


def classify_with_keywords(
    title: str,
    content: str,
    registry: AgentRegistry | None = None,
) -> TopicClassification:
    registry = registry or get_agent_registry()
    combined = f"{title}\n{content}".lower()
    scores = {
        topic: sum(1 for keyword in agent.keywords if keyword in combined)
        for topic, agent in registry.agents.items()
        if topic != "general"
    }
    if not scores:
        return TopicClassification("general", 0.25, "No specialized agents are enabled.", "keyword_fallback")
    topic, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return TopicClassification("general", 0.25, "No subject-specific keywords matched.", "keyword_fallback")
    confidence = min(0.95, 0.5 + score * 0.1)
    return TopicClassification(topic, confidence, f"Matched {score} subject keyword(s).", "keyword_fallback")


def parse_classification_payload(
    payload: dict[str, Any],
    registry: AgentRegistry | None = None,
) -> TopicClassification:
    topic = normalize_topic(str(payload.get("topic", "general")), registry)
    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    reason = str(payload.get("reason", "Classified by the routing model.")).strip()
    return TopicClassification(topic, confidence, reason, "llm")


class OpenAITopicClassifier:
    def __init__(self, config: Config, registry: AgentRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or get_agent_registry()
        self.provider = OpenAIProvider(config)
        self.session = self.provider.session

    def usage_totals(self) -> UsageTotals:
        totals = UsageTotals()
        totals.add(self.provider.usage_totals)
        return totals

    def classify(
        self,
        assignment: Assignment,
        attachment_texts: list[str],
        attachment_image_urls: list[str],
    ) -> TopicClassification:
        text = "\n\n".join(value.strip() for value in attachment_texts if value.strip())
        topic_descriptions = "\n".join(
            f"- {topic}: {agent.display_name}"
            for topic, agent in self.registry.agents.items()
        )
        prompt = (
            "Classify this university assignment into exactly one of the configured topics below. "
            "Inspect the PDF page images, not only the title. Select general only when no specialist fits.\n"
            f"Configured topics:\n{topic_descriptions}\n"
            "Return a JSON object with topic, confidence (0 to 1), and a short reason.\n\n"
            f"Course: {assignment.course_name}\nAssignment: {assignment.title}\n"
            f"Description: {assignment.intro_html}\nExtracted attachment text:\n{text[:12000]}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_url in attachment_image_urls:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        try:
            response = self.provider.generate(
                model=self.config.openai_model,
                system_content="You route assignments to specialist academic agents.",
                user_content=content,
                temperature=0,
                reasoning_effort=self.config.openai_reasoning_effort,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "assignment_classification",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "confidence": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                            "required": ["topic", "confidence", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
            return parse_classification_payload(json.loads(response.content), self.registry)
        except (
            requests.RequestException, WorkflowError, KeyError, IndexError,
            TypeError, ValueError, json.JSONDecodeError,
        ):
            return classify_with_keywords(
                f"{assignment.course_name} {assignment.title} {assignment.intro_html}",
                text,
                self.registry,
            )
