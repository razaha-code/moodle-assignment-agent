import datetime as dt
import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .agent_loader import get_agent_registry
from .classifier import OpenAITopicClassifier, TopicClassification, classify_with_keywords
from .config import bootstrap_moodle_token, load_config, load_env_file
from .llm import LLMAnswerGenerator
from .manifest import manifest_text
from .models import UsageTotals, WorkflowError, WorkflowStageError
from .moodle_client import MoodleClient
from .output import (
    build_assignment_output_dir,
    ensure_output_dir,
    render_html_to_pdf,
    sanitize_filename,
    strip_html_to_text,
    write_routing_report,
    write_html_file,
    write_text_file,
)
from .pdf_reader import collect_attachment_context
from .selection import evaluate_assignment, resolve_selection_window, resolve_target_date
from .subject_agents import get_subject_agent


logger = logging.getLogger(__name__)


def run_stage(stage: str, operation, usage_provider=None):
    try:
        return operation()
    except WorkflowStageError as exc:
        if callable(usage_provider) and not hasattr(exc, "llm_usage"):
            exc.llm_usage = usage_provider().to_dict()
        raise
    except Exception as exc:
        wrapped = WorkflowStageError(stage, exc)
        if callable(usage_provider):
            wrapped.llm_usage = usage_provider().to_dict()
        raise wrapped from exc


def write_document_bundle(
    assignment_dir: Path,
    assignment_title: str,
    basename: str,
    body_html: str,
    preview_image_paths: list[str],
    *,
    output_style: str = "standard",
    paper_style: str = "plain",
    synthetic_glyph_variation: bool | None = None,
    handwriting_font: str = "personal",
) -> tuple[Path, Path, Path, Path, str]:
    text_path = assignment_dir / f"{basename}.txt"
    html_path = assignment_dir / f"{basename}.html"
    json_path = assignment_dir / f"{basename}.json"
    pdf_name = "submission.pdf" if basename == "answer" else f"{basename}.pdf"
    pdf_path = assignment_dir / pdf_name

    plain_text = strip_html_to_text(body_html)
    write_text_file(text_path, plain_text)
    write_html_file(
        html_path,
        assignment_title,
        body_html,
        preview_image_paths=preview_image_paths,
        output_style=output_style,
        paper_style=paper_style,
        synthetic_glyph_variation=synthetic_glyph_variation,
        handwriting_font=handwriting_font,
    )
    pdf_engine = render_html_to_pdf(
        html_path,
        pdf_path,
        assignment_title,
        plain_text,
        output_style=output_style,
    )
    return text_path, html_path, json_path, pdf_path, pdf_engine


def process_assignment(args, config, reference_date: dt.date, assignment) -> dict[str, Any]:
    # requests.Session is not shared between workers. Each assignment gets isolated API clients.
    moodle = MoodleClient(config)
    generator = LLMAnswerGenerator(config)
    agent_registry = get_agent_registry()
    classifier = OpenAITopicClassifier(config, agent_registry)

    assignment_dir = build_assignment_output_dir(
        reference_date,
        assignment,
        run_label=getattr(args, "run_label", None),
    )
    attachment_context = run_stage(
        "attachment_processing",
        lambda: collect_attachment_context(moodle, assignment, assignment_dir),
    )
    manifest, manifest_path = run_stage(
        "manifest_extraction",
        lambda: generator.load_or_build_manifest(
            assignment,
            attachment_context,
            assignment_dir,
            rebuild=bool(getattr(args, "rebuild_manifest", False)),
        ),
        usage_provider=generator.usage_totals,
    )
    routing_text = manifest_text(manifest)
    keyword_route = classify_with_keywords(
        f"{assignment.course_name} {assignment.title} {assignment.intro_html}",
        routing_text,
        agent_registry,
    )
    if keyword_route.topic != "general" and keyword_route.confidence >= 0.6:
        classification = TopicClassification(
            keyword_route.topic,
            keyword_route.confidence,
            keyword_route.reason,
            "keyword_manifest",
        )
    else:
        classification = run_stage(
            "classification",
            lambda: classifier.classify(
                assignment,
                [*attachment_context.texts, routing_text],
                [],
            ),
        )
    subject_agent = get_subject_agent(classification.topic, agent_registry)
    subject_reasoning_effort = getattr(subject_agent, "reasoning_effort", None)
    if not isinstance(subject_reasoning_effort, str):
        subject_reasoning_effort = config.openai_reasoning_effort
    subject_output_style = getattr(subject_agent, "output_style", "standard")
    if subject_output_style not in {"standard", "ipad_clean"}:
        subject_output_style = "standard"
    subject_paper_style = getattr(subject_agent, "paper_style", "plain")
    if subject_paper_style not in {"plain", "grid", "ruled"}:
        subject_paper_style = "plain"
    synthetic_glyph_variation = subject_output_style == "ipad_clean"
    handwriting_font = getattr(args, "handwriting_font", "personal")
    if handwriting_font not in {"personal", "gveret"}:
        handwriting_font = "personal"
    primary_document = run_stage(
        "generation",
        lambda: subject_agent.generate_from_manifest(
            generator,
            assignment,
            manifest,
            args.workflow_mode,
            checkpoint_path=assignment_dir / "generation_checkpoint.json",
            manifest_path=manifest_path,
            page_image_paths=attachment_context.preview_image_paths,
        ),
        usage_provider=generator.usage_totals,
    )
    output_html = primary_document.html

    if args.workflow_mode == "instructions":
        primary_basename = "instructions"
        primary_title = assignment.title
    else:
        primary_basename = "answer"
        primary_title = assignment.title

    text_path, html_path, metadata_path, pdf_path, pdf_engine = write_document_bundle(
        assignment_dir,
        primary_title,
        primary_basename,
        output_html,
        attachment_context.preview_image_paths,
        output_style=subject_output_style,
        paper_style=subject_paper_style,
        synthetic_glyph_variation=synthetic_glyph_variation,
        handwriting_font=handwriting_font,
    )
    primary_coverage_path = assignment_dir / f"{primary_basename}_coverage.json"
    primary_coverage_path.write_text(
        json.dumps(primary_document.coverage_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    solution_text_path = None
    solution_html_path = None
    solution_json_path = None
    solution_pdf_path = None
    solution_pdf_engine = None
    solution_coverage_path = None
    solution_document = None

    solver_stage = bool(getattr(args, "solver_stage", False))
    if args.workflow_mode == "instructions" and solver_stage:
        instructions_text = text_path.read_text(encoding="utf-8")
        solution_document = run_stage(
            "solver_generation",
            lambda: generator.generate_specialist_document_from_manifest(
                assignment,
                manifest,
                "solve",
                subject_agent.display_name,
                f"{subject_agent.instructions}\nStage-1 instructions:\n{instructions_text}",
                checkpoint_path=assignment_dir / "solution_generation_checkpoint.json",
                manifest_path=manifest_path,
                page_image_paths=attachment_context.preview_image_paths,
                provider_name=subject_agent.provider,
                model=subject_agent.model,
                temperature=subject_agent.temperature,
                reasoning_effort=subject_reasoning_effort,
            ),
            usage_provider=generator.usage_totals,
        )
        solution_html = solution_document.html
        (
            solution_text_path,
            solution_html_path,
            solution_json_path,
            solution_pdf_path,
            solution_pdf_engine,
        ) = write_document_bundle(
            assignment_dir,
            f"{assignment.title} Solution",
            "solution",
            solution_html,
            attachment_context.preview_image_paths,
            output_style=subject_output_style,
            paper_style=subject_paper_style,
            synthetic_glyph_variation=synthetic_glyph_variation,
            handwriting_font=handwriting_font,
        )
        solution_coverage_path = assignment_dir / "solution_coverage.json"
        solution_coverage_path.write_text(
            json.dumps(solution_document.coverage_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    final_document = solution_document or primary_document
    coverage_status = "complete" if final_document.complete else "incomplete"
    usage = generator.usage_totals()
    classifier_usage_provider = getattr(classifier, "usage_totals", None)
    if callable(classifier_usage_provider):
        classifier_usage = classifier_usage_provider()
        if isinstance(classifier_usage, UsageTotals):
            usage.add(classifier_usage)

    result: dict[str, Any] = {
        "status": "completed" if final_document.complete else "incomplete",
        "course": assignment.course_name,
        "assignment": assignment.title,
        "assignment_id": assignment.assignment_id,
        "due_at": assignment.due_at.isoformat() if assignment.due_at else None,
        "already_submitted": assignment.already_submitted,
        "submission_status_before": assignment.submission_status,
        "workflow_mode": args.workflow_mode,
        "classified_topic": classification.topic,
        "classification_confidence": classification.confidence,
        "classification_reason": classification.reason,
        "classification_method": classification.method,
        "subject_agent": subject_agent.display_name,
        "llm_provider": subject_agent.provider,
        "llm_model": subject_agent.model or config.openai_model,
        "llm_temperature": subject_agent.temperature,
        "llm_reasoning_effort": subject_reasoning_effort,
        "output_style": subject_output_style,
        "paper_style": subject_paper_style,
        "handwriting_font": handwriting_font,
        "synthetic_glyph_variation": synthetic_glyph_variation,
        "source_files": attachment_context.saved_files,
        "attachment_text_count": len(attachment_context.texts),
        "attachment_image_count": len(attachment_context.image_urls),
        "preview_image_count": len(attachment_context.preview_image_paths),
        "manifest_path": str(manifest_path),
        "manifest_from_cache": manifest.from_cache,
        "manifest_schema_version": manifest.schema_version,
        "manifest_retry_count": manifest.manifest_retry_count,
        "retried_pages": list(manifest.retried_pages),
        "vision_request_count": manifest.vision_request_count,
        "fallback_crop_count": manifest.fallback_crop_count,
        "resumed_from_checkpoint": final_document.resumed_from_checkpoint,
        "checkpoint_path": final_document.checkpoint_path,
        "llm_usage": usage.to_dict(),
        "answer_text_path": str(text_path),
        "answer_html_path": str(html_path),
        "pdf_path": str(pdf_path),
        "pdf_engine": pdf_engine,
        "html_is_canonical": True,
        "coverage_status": coverage_status,
        "coverage_expected_count": len(final_document.inventory),
        "coverage_answered_count": len(final_document.answers),
        "coverage_missing_ids": final_document.missing_ids,
        "coverage_path": str(primary_coverage_path),
        "solver_stage_enabled": solver_stage,
        "solution_text_path": str(solution_text_path) if solution_text_path else None,
        "solution_html_path": str(solution_html_path) if solution_html_path else None,
        "solution_json_path": str(solution_json_path) if solution_json_path else None,
        "solution_pdf_path": str(solution_pdf_path) if solution_pdf_path else None,
        "solution_pdf_engine": solution_pdf_engine,
        "solution_coverage_path": str(solution_coverage_path) if solution_coverage_path else None,
        "submitted_now": False,
    }
    if pdf_engine.endswith("nonlatin-risk-fallback"):
        result["warning"] = "PDF fell back to the basic writer and may not preserve non-Latin text correctly."

    if args.submit and final_document.complete:
        submission_result = moodle.submit_assignment(assignment, pdf_path, strip_html_to_text(output_html))
        result["submitted_now"] = True
        result["submission_result"] = {
            "draft_item_id": submission_result["draft_item_id"],
            "submission_status_after": submission_result["submission_status"],
        }
    elif args.submit:
        result["submission_blocked_reason"] = (
            "Submission blocked because the generated document is incomplete. "
            f"Missing question IDs: {', '.join(final_document.missing_ids)}"
        )

    metadata_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if solution_json_path is not None:
        solution_json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def process_assignment_safely(args, config, reference_date: dt.date, assignment) -> dict[str, Any]:
    try:
        return process_assignment(args, config, reference_date, assignment)
    except Exception as exc:
        logger.exception("Subject-agent worker failed for assignment %s", assignment.assignment_id)
        cause = exc.cause if isinstance(exc, WorkflowStageError) else exc
        failure_model = getattr(config, "openai_model", None)
        failure_reasoning = getattr(config, "openai_reasoning_effort", None)
        return {
            "status": "error",
            "course": assignment.course_name,
            "assignment": assignment.title,
            "assignment_id": assignment.assignment_id,
            "due_at": assignment.due_at.isoformat() if assignment.due_at else None,
            "classified_topic": "unclassified",
            "subject_agent": None,
            "llm_model": failure_model if isinstance(failure_model, str) else None,
            "llm_reasoning_effort": failure_reasoning if isinstance(failure_reasoning, str) else None,
            "handwriting_font": getattr(args, "handwriting_font", "personal"),
            "submitted_now": False,
            "failed_stage": exc.stage if isinstance(exc, WorkflowStageError) else "unknown",
            "openai_status_code": getattr(cause, "status_code", None),
            "openai_error_type": getattr(cause, "error_type", "") or None,
            "openai_error_code": getattr(cause, "error_code", "") or None,
            "openai_request_id": getattr(cause, "request_id", "") or None,
            "manifest_validation_path": getattr(cause, "manifest_validation_path", None),
            "manifest_raw_path": getattr(cause, "manifest_raw_path", None),
            "manifest_retry_raw_path": getattr(cause, "manifest_retry_raw_path", None),
            "retried_pages": list(getattr(cause, "retried_pages", ()) or ()),
            "llm_usage": getattr(exc, "llm_usage", UsageTotals().to_dict()),
            "error": str(exc),
        }


def run_workflow(args) -> dict[str, Any]:
    env_path = Path(args.env_file)
    load_env_file(env_path)
    if args.bootstrap_token and not os.environ.get("MOODLE_TOKEN", "").strip():
        bootstrap_moodle_token(env_path, args.save_token)
    if args.workflow_mode == "instructions" and args.submit:
        raise WorkflowError("Instructions mode does not support --submit. Use workflow mode solve to submit.")
    if bool(getattr(args, "include_closed", False)) and args.submit:
        raise WorkflowError("--include-closed is for local drafts only and cannot be combined with --submit.")

    config = load_config()
    reference_date = resolve_target_date(config, args.date)
    selection_window = resolve_selection_window(reference_date, args.selection_window)
    output_dir = ensure_output_dir(reference_date)
    raw_run_label = getattr(args, "run_label", None)
    run_label = sanitize_filename(raw_run_label) if isinstance(raw_run_label, str) and raw_run_label.strip() else None
    now = dt.datetime.now(ZoneInfo(config.timezone))

    moodle = MoodleClient(config)
    assignments = moodle.get_assignments()
    moodle.mark_submission_status(assignments)
    relevant: list[Any] = []
    skipped: list[dict[str, Any]] = []

    for assignment in assignments:
        decision = evaluate_assignment(
            assignment,
            selection_window,
            args.include_submitted,
            now=now,
            include_closed=bool(getattr(args, "include_closed", False)),
        )
        if decision["relevant"]:
            relevant.append(assignment)
        else:
            skipped.append(
                {
                    "course": assignment.course_name,
                    "assignment": assignment.title,
                    "assignment_id": assignment.assignment_id,
                    "due_at": assignment.due_at.isoformat() if assignment.due_at else None,
                    "reason": decision["reason"],
                }
            )

    relevant.sort(key=lambda item: item.due_at or now)
    if args.max_assignments is not None:
        relevant = relevant[: args.max_assignments]

    requested_workers = max(1, int(getattr(args, "max_workers", 3)))
    worker_count = min(requested_workers, len(relevant)) if relevant else 0
    if worker_count > 0:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="subject-agent") as executor:
            results = list(
                executor.map(
                    lambda assignment: process_assignment_safely(args, config, reference_date, assignment),
                    relevant,
                )
            )
    else:
        results = []

    topic_counts = Counter(result["classified_topic"] for result in results)
    error_count = sum(1 for result in results if result.get("status") == "error")
    incomplete_count = sum(1 for result in results if result.get("status") == "incomplete")
    aggregate_usage = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "request_count": 0,
        "vision_request_count": 0,
        "fallback_crop_count": 0,
        "manifest_retry_count": 0,
        "estimated_cost_usd": 0.0,
    }
    for result in results:
        usage = result.get("llm_usage") or {}
        for key in aggregate_usage:
            aggregate_usage[key] += usage.get(key, 0) or 0
    aggregate_usage["estimated_cost_usd"] = round(aggregate_usage["estimated_cost_usd"], 6)

    summary = {
        "generated_at": now.isoformat(),
        "reference_date": reference_date.isoformat(),
        "selection_window": {
            "mode": selection_window.mode,
            "start_date": selection_window.start_date.isoformat(),
            "end_date": selection_window.end_date.isoformat(),
        },
        "submit_mode": args.submit,
        "run_label": run_label,
        "worker_count": worker_count,
        "topic_counts": dict(topic_counts),
        "error_count": error_count,
        "incomplete_count": incomplete_count,
        "llm_usage": aggregate_usage,
        "assignment_count": len(results),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "results": results,
    }

    report_basename = f"routing_report_{run_label}" if run_label else "routing_report"
    routing_json_path, routing_html_path = write_routing_report(
        output_dir,
        summary,
        basename=report_basename,
    )
    summary["routing_report_json_path"] = str(routing_json_path)
    summary["routing_report_html_path"] = str(routing_html_path)

    summary_name = f"summary_{run_label}.json" if run_label else "summary.json"
    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
