from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from .models import Config, UsageTotals, WorkflowError
from .openai_http import post_openai_json


@dataclass(frozen=True)
class LLMResponse:
    content: str
    finish_reason: str
    usage: UsageTotals = field(default_factory=UsageTotals)
    request_id: str = ""


class LLMProvider(Protocol):
    name: str

    def generate(
        self,
        *,
        model: str,
        system_content: str,
        user_content: list[dict[str, Any]],
        temperature: float,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...


class OpenAIProvider:
    name = "openai"

    def __init__(self, config: Config) -> None:
        self.api_key = config.openai_api_key
        self.session = requests.Session()
        self.usage_totals = UsageTotals()

    def generate(
        self,
        *,
        model: str,
        system_content: str,
        user_content: list[dict[str, Any]],
        temperature: float,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        }
        if model.startswith("gpt-5.6"):
            payload["reasoning_effort"] = reasoning_effort or "medium"
        else:
            payload["temperature"] = temperature
        if response_format is not None:
            payload["response_format"] = response_format
        if max_tokens is not None:
            if model.startswith("gpt-5.6"):
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
        response = post_openai_json(
            self.session,
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout=120,
        )
        response_payload = response.json()
        choices = response_payload.get("choices") or []
        if not choices:
            raise WorkflowError(f"OpenAI model {model!r} did not return any answer choices.")
        content = choices[0].get("message", {}).get("content", "")
        if not isinstance(content, str):
            content = ""
        raw_usage = response_payload.get("usage") or {}
        prompt_details = raw_usage.get("prompt_tokens_details") or {}
        input_tokens = int(raw_usage.get("prompt_tokens") or 0)
        cached_tokens = int(prompt_details.get("cached_tokens") or 0)
        output_tokens = int(raw_usage.get("completion_tokens") or 0)
        completion_details = raw_usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
        is_vision = any(item.get("type") == "image_url" for item in user_content)
        uncached_tokens = max(0, input_tokens - cached_tokens)
        if model.startswith("gpt-4o-mini"):
            estimated_cost = (
                uncached_tokens * 0.15 / 1_000_000
                + cached_tokens * 0.075 / 1_000_000
                + output_tokens * 0.60 / 1_000_000
            )
        elif model.startswith("gpt-5.6-terra"):
            large_context_multiplier = input_tokens > 272_000
            input_rate = 2.0 * (2.0 if large_context_multiplier else 1.0)
            cached_rate = 0.20 * (2.0 if large_context_multiplier else 1.0)
            output_rate = 12.0 * (1.5 if large_context_multiplier else 1.0)
            estimated_cost = (
                uncached_tokens * input_rate / 1_000_000
                + cached_tokens * cached_rate / 1_000_000
                + output_tokens * output_rate / 1_000_000
            )
        else:
            estimated_cost = 0.0
        usage = UsageTotals(
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            request_count=1,
            vision_request_count=1 if is_vision else 0,
            estimated_cost_usd=estimated_cost,
        )
        self.usage_totals.add(usage)
        return LLMResponse(
            content=content,
            finish_reason=str(choices[0].get("finish_reason") or "unknown"),
            usage=usage,
            request_id=response.headers.get("x-request-id", ""),
        )


class LLMProviderRegistry:
    def __init__(self, config: Config) -> None:
        self.providers: dict[str, LLMProvider] = {
            "openai": OpenAIProvider(config),
        }

    @property
    def supported_providers(self) -> tuple[str, ...]:
        return tuple(self.providers)

    def get(self, provider_name: str) -> LLMProvider:
        normalized = (provider_name or "").strip().lower()
        provider = self.providers.get(normalized)
        if provider is None:
            supported = ", ".join(self.supported_providers)
            raise WorkflowError(
                f"Unsupported LLM provider {provider_name!r}. Supported providers: {supported}."
            )
        return provider

    def usage(self) -> UsageTotals:
        totals = UsageTotals()
        for provider in self.providers.values():
            provider_usage = getattr(provider, "usage_totals", None)
            if isinstance(provider_usage, UsageTotals):
                totals.add(provider_usage)
        return totals
