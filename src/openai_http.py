import os
import re
import threading
import time
from typing import Any, Callable

import requests

from .models import WorkflowError


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_request_lock = threading.Lock()
_last_request_at = 0.0
_blocked_until = 0.0


class OpenAIRequestError(WorkflowError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str = "",
        error_code: str = "",
        request_id: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.request_id = request_id


def _error_details(response: requests.Response) -> tuple[str, str, str]:
    try:
        error = response.json().get("error") or {}
    except (ValueError, AttributeError):
        error = {}
    if not isinstance(error, dict):
        error = {}
    return (
        str(error.get("message") or response.text or "OpenAI request failed.").strip(),
        str(error.get("type") or "").strip(),
        str(error.get("code") or "").strip(),
    )


def _parse_duration(value: str) -> float | None:
    value = value.strip().lower()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    matches = re.findall(r"([0-9]*\.?[0-9]+)(ms|s|m|h)", value)
    if not matches:
        return None
    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(amount) * multipliers[unit] for amount, unit in matches)


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = _parse_duration(response.headers.get("Retry-After", ""))
        if retry_after is not None:
            return min(30.0, retry_after)
        reset_values = [
            _parse_duration(response.headers.get("x-ratelimit-reset-requests", "")),
            _parse_duration(response.headers.get("x-ratelimit-reset-tokens", "")),
        ]
        usable_resets = [value for value in reset_values if value is not None]
        if usable_resets:
            return min(30.0, max(usable_resets))
    return min(15.0, float(2 ** (attempt - 1)))


def _pace_requests(sleep_fn: Callable[[float], None]) -> None:
    global _last_request_at
    try:
        minimum_interval = max(0.0, float(os.environ.get("OPENAI_MIN_REQUEST_INTERVAL_SECONDS", "0.25")))
    except ValueError:
        minimum_interval = 0.25
    with _request_lock:
        now = time.monotonic()
        remaining = max(
            minimum_interval - (now - _last_request_at),
            _blocked_until - now,
        )
        if remaining > 0:
            sleep_fn(remaining)
        _last_request_at = time.monotonic()


def _observe_rate_limit_headers(response: requests.Response) -> None:
    global _blocked_until
    try:
        remaining = int(response.headers.get("x-ratelimit-remaining-tokens", ""))
        reserve = max(0, int(os.environ.get("OPENAI_TOKEN_SAFETY_RESERVE", "30000")))
    except ValueError:
        return
    if remaining >= reserve:
        return
    reset = _parse_duration(response.headers.get("x-ratelimit-reset-tokens", ""))
    if reset is None:
        return
    with _request_lock:
        _blocked_until = max(_blocked_until, time.monotonic() + min(60.0, reset))


def post_openai_json(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int = 120,
    max_attempts: int = 4,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> requests.Response:
    """POST to OpenAI with pacing and bounded retries for transient failures."""
    attempts = max(1, max_attempts)
    last_network_error: requests.RequestException | None = None

    for attempt in range(1, attempts + 1):
        _pace_requests(sleep_fn)
        try:
            response = session.post(url, headers=headers, json=payload, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_network_error = exc
            if attempt < attempts:
                sleep_fn(_retry_delay(None, attempt))
                continue
            raise OpenAIRequestError(
                f"OpenAI network request failed after {attempts} attempts: {exc}"
            ) from exc

        if response.status_code < 400:
            _observe_rate_limit_headers(response)
            return response

        message, error_type, error_code = _error_details(response)
        request_id = response.headers.get("x-request-id", "")
        quota_exhausted = "insufficient_quota" in {error_type, error_code}
        retryable = response.status_code in _RETRYABLE_STATUS_CODES and not quota_exhausted

        if retryable and attempt < attempts:
            sleep_fn(_retry_delay(response, attempt))
            continue

        if quota_exhausted:
            explanation = (
                "OpenAI quota is exhausted. Check API billing, project credits, and the project's spend limit."
            )
        elif response.status_code == 429:
            explanation = f"OpenAI rate limit persisted after {attempts} attempts."
        else:
            explanation = f"OpenAI API returned HTTP {response.status_code}."
        details = f" {message}" if message else ""
        request_suffix = f" Request ID: {request_id}." if request_id else ""
        raise OpenAIRequestError(
            f"{explanation}{details}{request_suffix}",
            status_code=response.status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
        )

    raise OpenAIRequestError(f"OpenAI request failed: {last_network_error}")
