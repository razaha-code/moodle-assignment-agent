import os
from pathlib import Path
from typing import Any

from .models import Config, RuntimeDiagnostics, WorkflowError
from .moodle_client import request_moodle_token_from_env
from .output import OUTPUT_DIR, PLAYWRIGHT_AVAILABLE, REPORTLAB_AVAILABLE, find_reportlab_font_path
from .pdf_reader import PYMUPDF_AVAILABLE


DEFAULT_BASE_URL = "https://moodle.bgu.ac.il/moodle"
DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_REASONING_EFFORT = "medium"
SUPPORTED_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_MOODLE_SERVICE = "moodle_mobile_app"


def load_config() -> Config:
    moodle_base_url = os.environ.get("MOODLE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    moodle_service = os.environ.get("MOODLE_SERVICE", DEFAULT_MOODLE_SERVICE).strip() or DEFAULT_MOODLE_SERVICE
    moodle_token = os.environ.get("MOODLE_TOKEN", "").strip()
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    active_courses_raw = os.environ.get("MOODLE_ACTIVE_COURSE_IDS", "").strip()

    if not moodle_token:
        moodle_token = request_moodle_token_from_env(moodle_base_url, moodle_service)

    if not moodle_token:
        raise WorkflowError(
            "Missing Moodle authentication. Set MOODLE_TOKEN or provide MOODLE_USERNAME and MOODLE_PASSWORD."
        )
    if not openai_api_key:
        raise WorkflowError("Missing OPENAI_API_KEY environment variable.")

    active_course_ids = {
        int(value.strip())
        for value in active_courses_raw.split(",")
        if value.strip()
    }
    reasoning_effort = (
        os.environ.get("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip()
        or DEFAULT_REASONING_EFFORT
    )
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise WorkflowError(
            "OPENAI_REASONING_EFFORT must be one of: "
            + ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
        )

    return Config(
        moodle_base_url=moodle_base_url,
        moodle_token=moodle_token,
        moodle_service=moodle_service,
        openai_api_key=openai_api_key,
        openai_model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        timezone=os.environ.get("MOODLE_TIMEZONE", "Asia/Jerusalem").strip() or "Asia/Jerusalem",
        active_course_ids=active_course_ids,
        openai_reasoning_effort=reasoning_effort,
    )


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def collect_runtime_diagnostics(env_path: Path) -> RuntimeDiagnostics:
    return RuntimeDiagnostics(
        env_file_path=str(env_path),
        env_file_exists=env_path.exists(),
        openai_api_key_present=bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        moodle_token_present=bool(os.environ.get("MOODLE_TOKEN", "").strip()),
        moodle_username_present=bool(os.environ.get("MOODLE_USERNAME", "").strip()),
        moodle_password_present=bool(os.environ.get("MOODLE_PASSWORD", "").strip()),
        active_course_ids_present=bool(os.environ.get("MOODLE_ACTIVE_COURSE_IDS", "").strip()),
        reportlab_available=REPORTLAB_AVAILABLE,
        reportlab_font_path=str(find_reportlab_font_path()) if find_reportlab_font_path() else None,
        playwright_available=PLAYWRIGHT_AVAILABLE,
        output_dir=str(OUTPUT_DIR),
        token_bootstrap_available=(
            bool(os.environ.get("MOODLE_USERNAME", "").strip())
            and bool(os.environ.get("MOODLE_PASSWORD", "").strip())
        ),
        pymupdf_available=PYMUPDF_AVAILABLE,
    )


def upsert_env_value(env_path: Path, key: str, value: str) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines: list[str] = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def bootstrap_moodle_token(env_path: Path, save_token: bool) -> str:
    base_url = os.environ.get("MOODLE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    service = os.environ.get("MOODLE_SERVICE", DEFAULT_MOODLE_SERVICE).strip() or DEFAULT_MOODLE_SERVICE
    token = request_moodle_token_from_env(base_url, service)
    if token:
        os.environ["MOODLE_TOKEN"] = token
        if save_token:
            upsert_env_value(env_path, "MOODLE_TOKEN", token)
    return token


def diagnostics_to_dict(diagnostics: RuntimeDiagnostics) -> dict[str, Any]:
    return {
        "env_file_path": diagnostics.env_file_path,
        "env_file_exists": diagnostics.env_file_exists,
        "openai_api_key_present": diagnostics.openai_api_key_present,
        "moodle_token_present": diagnostics.moodle_token_present,
        "moodle_username_present": diagnostics.moodle_username_present,
        "moodle_password_present": diagnostics.moodle_password_present,
        "active_course_ids_present": diagnostics.active_course_ids_present,
        "reportlab_available": diagnostics.reportlab_available,
        "reportlab_font_path": diagnostics.reportlab_font_path,
        "playwright_available": diagnostics.playwright_available,
        "output_dir": diagnostics.output_dir,
        "token_bootstrap_available": diagnostics.token_bootstrap_available,
        "pymupdf_available": diagnostics.pymupdf_available,
        "ready_for_live_run": (
            diagnostics.openai_api_key_present
            and (diagnostics.moodle_token_present or diagnostics.token_bootstrap_available)
            and (diagnostics.playwright_available or diagnostics.reportlab_available)
        ),
        "next_step": determine_next_step(diagnostics),
    }


def determine_next_step(diagnostics: RuntimeDiagnostics) -> str:
    if not diagnostics.openai_api_key_present:
        return "Add OPENAI_API_KEY to your environment or .env file."
    if not diagnostics.playwright_available:
        return "Install Playwright and Chromium with python -m pip install -r requirements.txt and python -m playwright install chromium."
    if diagnostics.moodle_token_present:
        return "Run python moodle_agent.py --date YYYY-MM-DD to test a dry run."
    if diagnostics.token_bootstrap_available:
        return "Run python moodle_agent.py --bootstrap-token --save-token to fetch and store a Moodle token."
    return "Add MOODLE_TOKEN or provide MOODLE_USERNAME and MOODLE_PASSWORD."
