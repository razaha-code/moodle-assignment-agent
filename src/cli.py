import argparse
import os
from pathlib import Path

from .config import (
    DEFAULT_ENV_FILE,
    bootstrap_moodle_token,
    collect_runtime_diagnostics,
    diagnostics_to_dict,
    load_env_file,
)
from .output import print_json_output
from .workflow import run_workflow


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch today's Moodle assignments, draft answers, export PDFs, and optionally submit them."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Reference date in YYYY-MM-DD format. Defaults to today in MOODLE_TIMEZONE.",
    )
    parser.add_argument(
        "--workflow-mode",
        choices=["instructions", "solve"],
        default="instructions",
        help="Generate solving instructions from the assignment PDF or generate a submission answer. Default: instructions.",
    )
    parser.add_argument(
        "--solver-stage",
        action="store_true",
        help="After generating instructions, run a second solver-agent stage that uses the instructions and source PDF to draft a solution.",
    )
    parser.add_argument(
        "--selection-window",
        choices=["week", "today"],
        default="week",
        help="Choose assignments due today or due by the end of the current week. Default: week.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Upload the generated PDF back to Moodle after drafting it.",
    )
    parser.add_argument(
        "--include-submitted",
        action="store_true",
        help="Include assignments that already appear as submitted.",
    )
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Allow generating local drafts for assignments whose submission window is closed. Cannot be combined with --submit.",
    )
    parser.add_argument(
        "--max-assignments",
        type=int,
        default=None,
        help="Optional cap on how many due-today assignments to process.",
    )
    parser.add_argument(
        "--max-workers",
        type=positive_int,
        default=3,
        help="Maximum assignments processed concurrently by the subject-agent pool. Default: 3.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to a local env file with OPENAI_API_KEY, MOODLE_TOKEN, and related settings.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Print local readiness diagnostics instead of running the workflow.",
    )
    parser.add_argument(
        "--bootstrap-token",
        action="store_true",
        help="Fetch a Moodle token from MOODLE_USERNAME and MOODLE_PASSWORD before running.",
    )
    parser.add_argument(
        "--save-token",
        action="store_true",
        help="When bootstrapping a Moodle token, save it into the env file for future runs.",
    )
    parser.add_argument(
        "--keep-render-debug",
        action="store_true",
        help="Reserved render-debug flag for troubleshooting HTML-to-PDF output without changing the main workflow.",
    )
    parser.add_argument(
        "--handwriting-font",
        choices=["personal", "gveret"],
        default="personal",
        help="Font used by ipad_clean output: personal handwriting or the bundled Gveret font. Default: personal.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional label that creates separate assignment, summary, and routing-report artifacts instead of overwriting a previous run.",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Ignore a compatible cached document manifest and run the one-pass vision extraction again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_path = Path(args.env_file)
    load_env_file(env_path)

    if args.bootstrap_token and not os.environ.get("MOODLE_TOKEN", "").strip():
        bootstrap_moodle_token(env_path, args.save_token)

    if args.doctor:
        diagnostics = collect_runtime_diagnostics(env_path)
        print_json_output(diagnostics_to_dict(diagnostics))
        return

    try:
        summary = run_workflow(args)
    except Exception as exc:
        raise SystemExit(f"Workflow failed: {exc}") from exc

    print_json_output(summary)
