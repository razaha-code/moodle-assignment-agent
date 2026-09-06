import datetime as dt
import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from src import cli
from src import config as config_module
from src import output as output_module
from src import workflow as workflow_module
from src.agent_loader import AgentConfigError, load_agent_registry
from src.classifier import OpenAITopicClassifier, TopicClassification, classify_with_keywords, normalize_topic
from src.chemistry_render import compound_reference_html, smiles_to_svg
from src.llm import (
    LLMAnswerGenerator,
    canonical_top_level,
    is_acceptable_answer_html,
    is_substantive_answer_html,
    normalize_html_fragment,
)
from src.llm_providers import LLMProviderRegistry, LLMResponse, OpenAIProvider
from src.manifest import MANIFEST_SCHEMA_VERSION, load_manifest
from src.models import (
    Assignment,
    AttachmentContext,
    Config,
    DocumentManifest,
    GeneratedDocument,
    ManifestQuestion,
    QuestionAnswer,
    QuestionItem,
    SelectionWindow,
    UsageTotals,
    VisualEvidence,
    WorkflowError,
    WorkflowStageError,
)
from src.openai_http import OpenAIRequestError, post_openai_json
from src.pdf_reader import (
    PYMUPDF_AVAILABLE,
    attachment_is_pdf,
    attachment_is_solution_reference,
    collect_attachment_context,
    extract_pdf_text,
    render_pdf_to_image_data_urls,
    render_pdf_to_png_bytes,
)
from src.selection import evaluate_assignment, resolve_selection_window
from src.subject_agents import get_subject_agent
from scripts.build_handwriting_variants import build_variants

from fontTools.ttLib import TTFont


class MoodleWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timezone = ZoneInfo("Asia/Jerusalem")
        self.target_date = dt.date(2026, 6, 18)

    def make_assignment(self, **overrides):
        base = Assignment(
            course_id=123,
            course_name="Computational Models",
            assignment_id=456,
            title="Daily Homework",
            intro_html="<p>Solve the exercise.</p>",
            due_at=dt.datetime(2026, 6, 18, 17, 0, tzinfo=self.timezone),
            allow_submissions_from=None,
            cutoff_at=None,
            attachments=[],
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base

    def test_evaluate_assignment_accepts_due_today(self):
        assignment = self.make_assignment()
        selection_window = SelectionWindow(
            mode="today",
            start_date=self.target_date,
            end_date=self.target_date,
        )

        decision = evaluate_assignment(
            assignment,
            selection_window,
            include_submitted=False,
            now=dt.datetime(2026, 6, 18, 12, 0, tzinfo=self.timezone),
        )

        self.assertEqual({"relevant": True, "reason": "due_today"}, decision)

    def test_evaluate_assignment_rejects_closed_submission_window(self):
        assignment = self.make_assignment(
            cutoff_at=dt.datetime(2026, 6, 18, 10, 0, tzinfo=self.timezone),
        )
        selection_window = SelectionWindow(
            mode="today",
            start_date=self.target_date,
            end_date=self.target_date,
        )

        decision = evaluate_assignment(
            assignment,
            selection_window,
            include_submitted=False,
            now=dt.datetime(2026, 6, 18, 12, 0, tzinfo=self.timezone),
        )

        self.assertEqual("submission_window_closed", decision["reason"])
        self.assertFalse(decision["relevant"])

    def test_evaluate_assignment_can_include_closed_for_local_draft(self):
        assignment = self.make_assignment(
            cutoff_at=dt.datetime(2026, 6, 18, 10, 0, tzinfo=self.timezone),
        )
        selection_window = SelectionWindow(
            mode="today", start_date=self.target_date, end_date=self.target_date
        )

        decision = evaluate_assignment(
            assignment,
            selection_window,
            include_submitted=True,
            now=dt.datetime(2026, 6, 18, 12, 0, tzinfo=self.timezone),
            include_closed=True,
        )

        self.assertTrue(decision["relevant"])
        self.assertEqual("due_today_closed_included", decision["reason"])

    def test_evaluate_assignment_rejects_already_submitted_by_default(self):
        assignment = self.make_assignment(already_submitted=True)
        selection_window = SelectionWindow(
            mode="today",
            start_date=self.target_date,
            end_date=self.target_date,
        )

        decision = evaluate_assignment(
            assignment,
            selection_window,
            include_submitted=False,
            now=dt.datetime(2026, 6, 18, 12, 0, tzinfo=self.timezone),
        )

        self.assertEqual("already_submitted", decision["reason"])
        self.assertFalse(decision["relevant"])

    def test_evaluate_assignment_accepts_due_by_end_of_week(self):
        assignment = self.make_assignment(
            due_at=dt.datetime(2026, 6, 20, 18, 0, tzinfo=self.timezone),
        )
        selection_window = resolve_selection_window(dt.date(2026, 6, 18), "week")

        decision = evaluate_assignment(
            assignment,
            selection_window,
            include_submitted=False,
            now=dt.datetime(2026, 6, 18, 12, 0, tzinfo=self.timezone),
        )

        self.assertEqual({"relevant": True, "reason": "due_by_end_of_week"}, decision)

    def test_resolve_selection_window_week_ends_on_saturday(self):
        selection_window = resolve_selection_window(dt.date(2026, 6, 21), "week")
        self.assertEqual(dt.date(2026, 6, 21), selection_window.start_date)
        self.assertEqual(dt.date(2026, 6, 27), selection_window.end_date)

    def test_attachment_is_pdf_by_filename_or_content_type(self):
        self.assertTrue(attachment_is_pdf("work.pdf", "application/octet-stream"))
        self.assertTrue(attachment_is_pdf("work.bin", "application/pdf"))
        self.assertFalse(attachment_is_pdf("work.txt", "text/plain"))

    def test_wrap_answer_text_preserves_blank_lines(self):
        wrapped = output_module.wrap_answer_text("line one\n\nline two", width=10)
        self.assertEqual(["line one", "", "line two"], wrapped)

    def test_build_answer_pdf_writes_valid_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "answer.pdf"
            pdf_engine = output_module.build_answer_pdf(pdf_path, "Title", ["Line 1", "Line 2"])

            self.assertTrue(pdf_path.exists())
            content = pdf_path.read_bytes()
            self.assertTrue(content.startswith(b"%PDF"))
            self.assertIn(pdf_engine, {"basic", "reportlab", "basic-nonlatin-risk"})

    def test_build_answer_pdf_flags_non_latin_risk_without_reportlab(self):
        original_flag = output_module.REPORTLAB_AVAILABLE
        try:
            output_module.REPORTLAB_AVAILABLE = False
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = Path(temp_dir) / "answer.pdf"
                pdf_engine = output_module.build_answer_pdf(pdf_path, "כותרת", ["שלום עולם"])

                self.assertEqual("basic-nonlatin-risk", pdf_engine)
                self.assertTrue(pdf_path.exists())
        finally:
            output_module.REPORTLAB_AVAILABLE = original_flag

    def test_render_pdf_to_image_data_urls_returns_images_when_supported(self):
        if not PYMUPDF_AVAILABLE:
            self.assertEqual([], render_pdf_to_image_data_urls(b""))
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            output_module.build_answer_pdf(pdf_path, "Title", ["Line 1", "Line 2"])
            image_urls = render_pdf_to_image_data_urls(pdf_path.read_bytes(), max_pages=2)
            self.assertTrue(image_urls)
            self.assertTrue(image_urls[0].startswith("data:image/png;base64,"))

    def test_render_pdf_to_png_bytes_returns_bytes_when_supported(self):
        if not PYMUPDF_AVAILABLE:
            self.assertEqual([], render_pdf_to_png_bytes(b""))
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            output_module.build_answer_pdf(pdf_path, "Title", ["Line 1", "Line 2"])
            images = render_pdf_to_png_bytes(pdf_path.read_bytes(), max_pages=1)
            self.assertTrue(images)
            self.assertIsInstance(images[0], bytes)

    def test_extract_pdf_text_reads_all_text_pages_when_supported(self):
        if not PYMUPDF_AVAILABLE:
            self.assertEqual("", extract_pdf_text(b""))
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            output_module.build_answer_pdf(pdf_path, "Assignment", ["Question one", "Question two"])

            extracted = extract_pdf_text(pdf_path.read_bytes())

            self.assertIn("Question one", extracted)
            self.assertIn("Question two", extracted)

    def test_collect_attachment_context_parses_and_saves_c_source(self):
        assignment = self.make_assignment(
            attachments=[{"filename": "worker.c", "fileurl": "https://example.test/worker.c"}]
        )
        client = Mock()
        client.download_attachment_bytes.return_value = (
            b"#include <pthread.h>\nint main(void) { return 0; }\n",
            "text/x-c",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context = collect_attachment_context(client, assignment, Path(temp_dir))

            self.assertIn("pthread.h", context.texts[0])
            self.assertEqual("worker.c", context.saved_files[0]["filename"])
            self.assertTrue(Path(context.saved_files[0]["local_path"]).exists())

    def test_solution_pdf_is_saved_as_reference_but_not_sent_as_vision_page(self):
        if not PYMUPDF_AVAILABLE:
            self.skipTest("PyMuPDF is required for PDF attachment rendering")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment_pdf = root / "assignment.pdf"
            solution_pdf = root / "solution.pdf"
            output_module.build_answer_pdf(assignment_pdf, "HW", ["Question 1"])
            output_module.build_answer_pdf(solution_pdf, "Solution", ["Answer 1"])
            assignment = self.make_assignment(attachments=[
                {"filename": "HW6.pdf", "fileurl": "https://example.test/HW6.pdf"},
                {"filename": "HW6_Sol.pdf", "fileurl": "https://example.test/HW6_Sol.pdf"},
            ])
            client = Mock()
            client.download_attachment_bytes.side_effect = [
                (assignment_pdf.read_bytes(), "application/pdf"),
                (solution_pdf.read_bytes(), "application/pdf"),
            ]

            context = collect_attachment_context(client, assignment, root / "out")

        self.assertTrue(attachment_is_solution_reference("HW6_Sol.pdf"))
        self.assertFalse(attachment_is_solution_reference("HW6.pdf"))
        self.assertEqual(1, len(context.image_urls))
        self.assertEqual("assignment", context.saved_files[0]["source_role"])
        self.assertEqual("solution_reference", context.saved_files[1]["source_role"])
        self.assertEqual(0, context.saved_files[1]["image_count"])
        self.assertIn("REFERENCE SOLUTION PDF", context.texts[1])

    def test_load_env_file_sets_missing_variables_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=file-key\nMOODLE_TOKEN=file-token\nMOODLE_TIMEZONE=Asia/Jerusalem\n",
                encoding="utf-8",
            )

            previous_openai = os.environ.get("OPENAI_API_KEY")
            previous_token = os.environ.get("MOODLE_TOKEN")
            previous_timezone = os.environ.get("MOODLE_TIMEZONE")

            try:
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ["MOODLE_TOKEN"] = "existing-token"
                os.environ.pop("MOODLE_TIMEZONE", None)

                config_module.load_env_file(env_path)

                self.assertEqual("file-key", os.environ.get("OPENAI_API_KEY"))
                self.assertEqual("existing-token", os.environ.get("MOODLE_TOKEN"))
                self.assertEqual("Asia/Jerusalem", os.environ.get("MOODLE_TIMEZONE"))
            finally:
                if previous_openai is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = previous_openai

                if previous_token is None:
                    os.environ.pop("MOODLE_TOKEN", None)
                else:
                    os.environ["MOODLE_TOKEN"] = previous_token

                if previous_timezone is None:
                    os.environ.pop("MOODLE_TIMEZONE", None)
                else:
                    os.environ["MOODLE_TIMEZONE"] = previous_timezone

    def test_write_html_file_preserves_unicode_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            html_path = Path(temp_dir) / "answer.html"
            output_module.write_html_file(html_path, "כותרת", "<section><p>שלום</p><p>עולם</p></section>")
            content = html_path.read_text(encoding="utf-8")
            self.assertIn("כותרת", content)
            self.assertIn("שלום", content)
            self.assertIn('dir="rtl"', content)

    def test_build_rich_html_document_includes_mathjax_and_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "preview.png"
            image_path.write_bytes(b"fake-image")
            document = output_module.build_rich_html_document(
                "כימיה אורגנית",
                "<section><p>נפתור את \\(x^2 + y^2\\).</p><div class=\"math-block\">$$E=mc^2$$</div></section>",
                preview_image_paths=[str(image_path)],
            )
            self.assertIn("tex-svg.js", document)
            self.assertIn("\\(x^2 + y^2\\)", document)
            self.assertIn("$$E=mc^2$$", document)
            self.assertIn(image_path.resolve().as_uri(), document)

    def test_ipad_clean_document_embeds_handwriting_font_grid_and_math_fitting(self):
        document = output_module.build_rich_html_document(
            "מודלים חישוביים",
            "<p>נוכיח את הטענה.</p><p class=\"ltr\">English term</p>"
            "<div class=\"math-block\">$$x_1+\\cdots+x_n$$</div>",
            output_style="ipad_clean",
            paper_style="grid",
        )

        self.assertIn('class="output-ipad-clean paper-grid"', document)
        self.assertIn("data:font/woff2;base64,", document)
        self.assertIn("font-family: 'Handwriting Base'", document)
        self.assertIn('data-handwriting-font="personal"', document)
        self.assertIn("7mm 7mm", document)
        self.assertIn("fitDisplayMath", document)
        self.assertIn("math-scaled-to-page", document)
        self.assertIn("'Consolas', 'Courier New', monospace", document)

    def test_ipad_clean_embeds_four_deterministic_glyph_variants(self):
        document = output_module.build_rich_html_document(
            "אותיות חוזרות",
            "<p>שלום שלום שלום</p><code>קוד עברי</code><p class=\"ltr\">English</p>",
            output_style="ipad_clean",
            paper_style="grid",
        )

        for index in range(1, 5):
            self.assertIn(f"Handwriting Variant {index}", document)
            self.assertIn(f"glyph-variant-{index}", document)
        self.assertIn("new Intl.Segmenter('he'", document)
        self.assertIn("A-Za-z", document)
        self.assertIn("window.__personalLatinGlyphs = true", document)
        self.assertIn("data-synthetic-glyph-variation=\"true\"", document)
        self.assertIn("mjx-container, code, pre, .ltr", document)
        self.assertIn("window.__glyphVariationSeed", document)

    def test_ipad_clean_can_select_previous_gveret_font(self):
        personal_document = output_module.build_rich_html_document(
            "כתב אישי", "<p>בדיקה</p>", output_style="ipad_clean"
        )
        gveret_document = output_module.build_rich_html_document(
            "פונט קודם",
            "<p>בדיקה</p>",
            output_style="ipad_clean",
            handwriting_font="gveret",
        )

        self.assertIn('data-handwriting-font="personal"', personal_document)
        self.assertIn('data-handwriting-font="gveret"', gveret_document)
        self.assertNotEqual(personal_document, gveret_document)

    def test_cli_defaults_to_personal_handwriting_and_accepts_gveret(self):
        with patch("sys.argv", ["moodle_agent.py"]):
            args = cli.parse_args()
            self.assertEqual("personal", args.handwriting_font)
            self.assertIsNone(args.run_label)
        with patch("sys.argv", ["moodle_agent.py", "--handwriting-font", "gveret"]):
            self.assertEqual("gveret", cli.parse_args().handwriting_font)

    def test_synthetic_fonts_preserve_cmap_and_widths_but_change_hebrew_outlines(self):
        font_dir = Path(output_module.HANDWRITING_FONT_PATH).parent
        source = TTFont(font_dir / "GveretLevin-Regular.ttf", recalcTimestamp=False)
        variants = [TTFont(path, recalcTimestamp=False) for path in output_module.HANDWRITING_VARIANT_PATHS]
        source_cmap = source.getBestCmap()

        for codepoint in range(0x05D0, 0x05EB):
            glyph_name = source_cmap[codepoint]
            self.assertTrue(all(font.getBestCmap()[codepoint] == glyph_name for font in variants))
            self.assertTrue(
                all(
                    font["hmtx"].metrics[glyph_name] == source["hmtx"].metrics[glyph_name]
                    for font in variants
                )
            )

        for codepoint in range(0x05D0, 0x05EB):
            glyph_name = source_cmap[codepoint]
            outlines = []
            for font in variants:
                coordinates, _, _ = font["glyf"][glyph_name].getCoordinates(font["glyf"])
                outlines.append(tuple(coordinates))
            self.assertGreaterEqual(len(set(outlines)), 3)

    def test_synthetic_font_build_is_binary_deterministic(self):
        source_path = Path(output_module.HANDWRITING_FONT_PATH).with_suffix(".ttf")
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = build_variants(source_path, Path(first_dir))
            second = build_variants(source_path, Path(second_dir))
            first_hashes = [hashlib.sha256(path.read_bytes()).digest() for path in first]
            second_hashes = [hashlib.sha256(path.read_bytes()).digest() for path in second]
        self.assertEqual(first_hashes, second_hashes)

    def test_render_profile_rejects_unknown_style(self):
        with self.assertRaisesRegex(ValueError, "Unsupported output style"):
            output_module.build_rich_html_document(
                "Title", "<p>Answer</p>", output_style="pencil"
            )
        with self.assertRaisesRegex(ValueError, "Unsupported handwriting font"):
            output_module.build_rich_html_document(
                "Title", "<p>Answer</p>", handwriting_font="unknown"
            )

    def test_strip_html_to_text_removes_markup(self):
        text = output_module.strip_html_to_text("<section><h2>Title</h2><p>Hello <strong>world</strong></p></section>")
        self.assertEqual("Title\nHello world", text)

    def test_write_routing_report_records_classification_and_agent(self):
        summary = {
            "generated_at": "2026-08-19T12:00:00+03:00",
            "reference_date": "2026-08-19",
            "topic_counts": {"mathematics": 1},
            "assignment_count": 1,
            "error_count": 0,
            "skipped": [],
            "results": [
                {
                    "course": "Mathematics",
                    "assignment": "Homework <1>",
                    "assignment_id": 42,
                    "status": "completed",
                    "classified_topic": "mathematics",
                    "classification_confidence": 0.94,
                    "classification_method": "llm",
                    "classification_reason": "Calculus questions",
                    "subject_agent": "mathematics teacher",
                    "llm_provider": "openai",
                    "llm_model": "agent-model",
                    "llm_temperature": 0.2,
                    "output_style": "ipad_clean",
                    "paper_style": "grid",
                    "handwriting_font": "personal",
                    "synthetic_glyph_variation": True,
                    "submitted_now": False,
                    "pdf_path": "submission.pdf",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, html_path = output_module.write_routing_report(Path(temp_dir), summary)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            html_content = html_path.read_text(encoding="utf-8")

        self.assertEqual("mathematics", payload["results"][0]["classified_topic"])
        self.assertEqual("mathematics teacher", payload["results"][0]["subject_agent"])
        self.assertEqual("ipad_clean", payload["results"][0]["output_style"])
        self.assertEqual("grid", payload["results"][0]["paper_style"])
        self.assertEqual("personal", payload["results"][0]["handwriting_font"])
        self.assertTrue(payload["results"][0]["synthetic_glyph_variation"])
        self.assertIn("94%", html_content)
        self.assertIn("openai / agent-model", html_content)
        self.assertIn("ipad_clean / grid / font=personal", html_content)
        self.assertIn("personal glyph variation", html_content)
        self.assertIn("Homework &lt;1&gt;", html_content)

    def test_render_html_to_pdf_falls_back_without_playwright(self):
        original_flag = output_module.PLAYWRIGHT_AVAILABLE
        original_sync = output_module.sync_playwright
        try:
            output_module.PLAYWRIGHT_AVAILABLE = False
            output_module.sync_playwright = None
            with tempfile.TemporaryDirectory() as temp_dir:
                html_path = Path(temp_dir) / "answer.html"
                pdf_path = Path(temp_dir) / "answer.pdf"
                output_module.write_html_file(html_path, "כותרת", "<p>שלום</p>")
                engine = output_module.render_html_to_pdf(html_path, pdf_path, "כותרת", "שלום")
                self.assertTrue(pdf_path.exists())
                self.assertTrue(engine.endswith("-fallback"))
        finally:
            output_module.PLAYWRIGHT_AVAILABLE = original_flag
            output_module.sync_playwright = original_sync

    def test_ipad_clean_render_fails_explicitly_without_playwright(self):
        original_flag = output_module.PLAYWRIGHT_AVAILABLE
        original_sync = output_module.sync_playwright
        try:
            output_module.PLAYWRIGHT_AVAILABLE = False
            output_module.sync_playwright = None
            with tempfile.TemporaryDirectory() as temp_dir:
                html_path = Path(temp_dir) / "answer.html"
                pdf_path = Path(temp_dir) / "answer.pdf"
                output_module.write_html_file(
                    html_path,
                    "כותרת",
                    "<p>שלום</p>",
                    output_style="ipad_clean",
                    paper_style="grid",
                )
                with self.assertRaisesRegex(RuntimeError, "requires Playwright and Chromium"):
                    output_module.render_html_to_pdf(
                        html_path,
                        pdf_path,
                        "כותרת",
                        "שלום",
                        output_style="ipad_clean",
                    )
        finally:
            output_module.PLAYWRIGHT_AVAILABLE = original_flag
            output_module.sync_playwright = original_sync

    def test_collect_runtime_diagnostics_reflects_loaded_environment(self):
        previous_openai = os.environ.get("OPENAI_API_KEY")
        previous_token = os.environ.get("MOODLE_TOKEN")
        previous_courses = os.environ.get("MOODLE_ACTIVE_COURSE_IDS")
        previous_username = os.environ.get("MOODLE_USERNAME")
        previous_password = os.environ.get("MOODLE_PASSWORD")

        try:
            os.environ["OPENAI_API_KEY"] = "key"
            os.environ["MOODLE_TOKEN"] = "token"
            os.environ["MOODLE_ACTIVE_COURSE_IDS"] = "1,2,3"
            os.environ["MOODLE_USERNAME"] = "user"
            os.environ["MOODLE_PASSWORD"] = "pass"

            diagnostics = config_module.collect_runtime_diagnostics(Path(".env"))
            result = config_module.diagnostics_to_dict(diagnostics)

            self.assertTrue(result["openai_api_key_present"])
            self.assertTrue(result["moodle_token_present"])
            self.assertTrue(result["moodle_username_present"])
            self.assertTrue(result["moodle_password_present"])
            self.assertTrue(result["active_course_ids_present"])
            self.assertIn("playwright_available", result)
            self.assertIn("ready_for_live_run", result)
        finally:
            if previous_openai is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous_openai

            if previous_token is None:
                os.environ.pop("MOODLE_TOKEN", None)
            else:
                os.environ["MOODLE_TOKEN"] = previous_token

            if previous_courses is None:
                os.environ.pop("MOODLE_ACTIVE_COURSE_IDS", None)
            else:
                os.environ["MOODLE_ACTIVE_COURSE_IDS"] = previous_courses

            if previous_username is None:
                os.environ.pop("MOODLE_USERNAME", None)
            else:
                os.environ["MOODLE_USERNAME"] = previous_username

            if previous_password is None:
                os.environ.pop("MOODLE_PASSWORD", None)
            else:
                os.environ["MOODLE_PASSWORD"] = previous_password

    def test_request_moodle_token_parses_successful_response(self):
        mock_response = Mock()
        mock_response.json.return_value = {"token": "abc123"}
        mock_response.raise_for_status.return_value = None

        with patch("src.moodle_client.requests.post", return_value=mock_response) as mock_post:
            token = __import__("src.moodle_client", fromlist=["request_moodle_token"]).request_moodle_token(
                "https://moodle.example.com/moodle",
                "student",
                "secret",
                "moodle_mobile_app",
            )

        self.assertEqual("abc123", token)
        mock_post.assert_called_once()

    def test_upsert_env_value_updates_existing_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("OPENAI_API_KEY=old\nMOODLE_TOKEN=old-token\n", encoding="utf-8")

            config_module.upsert_env_value(env_path, "MOODLE_TOKEN", "new-token")

            content = env_path.read_text(encoding="utf-8")
            self.assertIn("MOODLE_TOKEN=new-token", content)
            self.assertNotIn("MOODLE_TOKEN=old-token", content)

    def test_bootstrap_moodle_token_saves_token_when_requested(self):
        previous_base_url = os.environ.get("MOODLE_BASE_URL")
        previous_service = os.environ.get("MOODLE_SERVICE")
        previous_username = os.environ.get("MOODLE_USERNAME")
        previous_password = os.environ.get("MOODLE_PASSWORD")
        previous_token = os.environ.get("MOODLE_TOKEN")

        try:
            os.environ["MOODLE_BASE_URL"] = "https://moodle.example.com/moodle"
            os.environ["MOODLE_SERVICE"] = "moodle_mobile_app"
            os.environ["MOODLE_USERNAME"] = "student"
            os.environ["MOODLE_PASSWORD"] = "secret"
            os.environ.pop("MOODLE_TOKEN", None)

            with tempfile.TemporaryDirectory() as temp_dir:
                env_path = Path(temp_dir) / ".env"
                with patch.object(config_module, "request_moodle_token_from_env", return_value="saved-token"):
                    token = config_module.bootstrap_moodle_token(env_path, save_token=True)

                self.assertEqual("saved-token", token)
                self.assertEqual("saved-token", os.environ.get("MOODLE_TOKEN"))
                self.assertIn("MOODLE_TOKEN=saved-token", env_path.read_text(encoding="utf-8"))
        finally:
            if previous_base_url is None:
                os.environ.pop("MOODLE_BASE_URL", None)
            else:
                os.environ["MOODLE_BASE_URL"] = previous_base_url

            if previous_service is None:
                os.environ.pop("MOODLE_SERVICE", None)
            else:
                os.environ["MOODLE_SERVICE"] = previous_service

            if previous_username is None:
                os.environ.pop("MOODLE_USERNAME", None)
            else:
                os.environ["MOODLE_USERNAME"] = previous_username

            if previous_password is None:
                os.environ.pop("MOODLE_PASSWORD", None)
            else:
                os.environ["MOODLE_PASSWORD"] = previous_password

            if previous_token is None:
                os.environ.pop("MOODLE_TOKEN", None)
            else:
                os.environ["MOODLE_TOKEN"] = previous_token

    def test_build_assignment_output_dir_naming(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_output_dir = output_module.OUTPUT_DIR
            output_module.OUTPUT_DIR = Path(temp_dir)
            try:
                assignment = self.make_assignment(course_name="Organic Chemistry", title="HW 8")
                assignment_dir = output_module.build_assignment_output_dir(dt.date(2026, 6, 21), assignment)
                self.assertTrue(assignment_dir.exists())
                self.assertEqual("2026-06-21_Organic_Chemistry_HW_8", assignment_dir.name)
                iteration_dir = output_module.build_assignment_output_dir(
                    dt.date(2026, 6, 21),
                    assignment,
                    run_label="personal handwriting v1",
                )
                self.assertEqual(
                    "2026-06-21_Organic_Chemistry_HW_8_personal_handwriting_v1",
                    iteration_dir.name,
                )
            finally:
                output_module.OUTPUT_DIR = original_output_dir

    def test_cli_main_is_callable(self):
        self.assertTrue(callable(cli.main))

    def test_keyword_classifier_routes_supported_subjects(self):
        chemistry = classify_with_keywords("Organic Chemistry", "SN2 substitution and stereochemistry")
        mathematics = classify_with_keywords("Calculus", "Find the derivative and prove the theorem")
        systems = classify_with_keywords("Systems Programming", "Implement pthread mutex synchronization on Linux")

        self.assertEqual("organic_chemistry", chemistry.topic)
        self.assertEqual("mathematics", mathematics.topic)
        self.assertEqual("systems_programming", systems.topic)
        self.assertEqual("mathematics", normalize_topic("math"))

    def test_subject_agent_registry_has_general_fallback(self):
        self.assertEqual("organic chemistry teacher", get_subject_agent("organic_chemistry").display_name)
        self.assertEqual("mathematics teacher", get_subject_agent("mathematics").display_name)
        self.assertEqual("systems programming teacher", get_subject_agent("systems_programming").display_name)
        self.assertEqual("general", get_subject_agent("unknown-topic").topic)

    def test_subject_agent_loads_llm_settings_from_toml(self):
        agent = get_subject_agent("mathematics")

        self.assertEqual("openai", agent.provider)
        self.assertEqual("gpt-5.6-terra", agent.model)
        self.assertEqual(0.2, agent.temperature)
        self.assertEqual("medium", agent.reasoning_effort)
        self.assertEqual("ipad_clean", agent.output_style)
        self.assertEqual("grid", agent.paper_style)

    def test_new_course_agents_are_discovered_and_classified_from_hebrew_titles(self):
        cases = {
            "ביוכימיה א": "biochemistry_a",
            "ביולוגיה מולקולרית": "molecular_biology",
            "אלגוריתמי אופטימיזציה, התאמה וחיפוש": "optimization_matching_search",
            "מודלים חישוביים": "computational_models",
        }

        for course_title, expected_topic in cases.items():
            with self.subTest(course_title=course_title):
                classification = classify_with_keywords(course_title, "")
                agent = get_subject_agent(expected_topic)
                self.assertEqual(expected_topic, classification.topic)
                self.assertEqual(expected_topic, agent.topic)
                self.assertTrue(agent.instructions)

    def test_agent_loader_discovers_new_toml_agent_without_code_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            agents_dir = Path(temp_dir)
            (agents_dir / "general.toml").write_text(
                'topic = "general"\ndisplay_name = "General"\ninstructions = "Fallback instructions"\n',
                encoding="utf-8",
            )
            (agents_dir / "physics.toml").write_text(
                'topic = "physics"\ndisplay_name = "Physics Teacher"\n'
                'aliases = ["phys"]\nkeywords = ["kinematics", "mechanics"]\n'
                'instructions = "Solve physics problems with units."\n',
                encoding="utf-8",
            )

            registry = load_agent_registry(agents_dir)
            classification = classify_with_keywords("Mechanics", "Solve the kinematics problem", registry)

        self.assertEqual(("general", "physics"), registry.topics)
        self.assertEqual("physics", registry.normalize_topic("phys"))
        self.assertEqual("Physics Teacher", registry.get("physics").display_name)
        self.assertEqual("physics", classification.topic)

    def test_agent_loader_requires_general_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "math.toml"
            path.write_text(
                'topic = "mathematics"\ndisplay_name = "Math"\ninstructions = "Solve carefully"\n',
                encoding="utf-8",
            )
            with self.assertRaises(AgentConfigError):
                load_agent_registry(Path(temp_dir))

    def test_agent_loader_rejects_invalid_temperature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "general.toml"
            path.write_text(
                'topic = "general"\ndisplay_name = "General"\n'
                'instructions = "Fallback"\ntemperature = 3.0\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AgentConfigError, "temperature must be between 0 and 2"):
                load_agent_registry(Path(temp_dir))

    def test_agent_loader_rejects_invalid_render_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "general.toml"
            path.write_text(
                'topic = "general"\ndisplay_name = "General"\n'
                'instructions = "Fallback"\noutput_style = "paper"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AgentConfigError, "output_style must be one of"):
                load_agent_registry(Path(temp_dir))

    def test_llm_classifier_reads_json_and_attaches_pdf_images(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="model",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        response = Mock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"topic":"systems_programming","confidence":0.91,"reason":"POSIX threads"}'
                }
            }]
        }
        classifier = OpenAITopicClassifier(config)
        classifier.session.post = Mock(return_value=response)

        result = classifier.classify(self.make_assignment(), ["pthread exercise"], ["data:image/png;base64,abc"])

        self.assertEqual("systems_programming", result.topic)
        self.assertEqual("llm", result.method)
        payload = classifier.session.post.call_args.kwargs["json"]
        user_content = payload["messages"][1]["content"]
        self.assertEqual("image_url", user_content[1]["type"])

    def test_openai_provider_uses_selected_model_and_temperature(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="default-model",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        response = Mock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "<p>answer</p>"}}],
            "usage": {
                "prompt_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 400},
                "completion_tokens": 200,
            },
        }
        provider = OpenAIProvider(config)
        provider.session.post = Mock(return_value=response)

        content = provider.generate(
            model="agent-model",
            system_content="system",
            user_content=[{"type": "text", "text": "question"}],
            temperature=0.65,
        )

        payload = provider.session.post.call_args.kwargs["json"]
        self.assertEqual("<p>answer</p>", content.content)
        self.assertEqual("agent-model", payload["model"])
        self.assertEqual(0.65, payload["temperature"])
        self.assertEqual(1000, content.usage.input_tokens)
        self.assertEqual(400, content.usage.cached_tokens)
        self.assertEqual(200, content.usage.output_tokens)
        self.assertEqual(1, provider.usage_totals.request_count)

    def test_openai_provider_uses_terra_reasoning_without_temperature_and_prices_long_context(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="gpt-5.6-terra",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
            openai_reasoning_effort="medium",
        )
        response = Mock(status_code=200, headers={})
        response.json.return_value = {
            "choices": [{"message": {"content": "{\"ok\":true}"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 300_000,
                "prompt_tokens_details": {"cached_tokens": 100_000},
                "completion_tokens": 1_000,
                "completion_tokens_details": {"reasoning_tokens": 800},
            },
        }
        provider = OpenAIProvider(config)
        provider.session.post = Mock(return_value=response)

        result = provider.generate(
            model="gpt-5.6-terra",
            system_content="system",
            user_content=[{"type": "text", "text": "question"}],
            temperature=0.7,
            reasoning_effort="medium",
            max_tokens=4000,
        )

        payload = provider.session.post.call_args.kwargs["json"]
        self.assertNotIn("temperature", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual("medium", payload["reasoning_effort"])
        self.assertEqual(4000, payload["max_completion_tokens"])
        self.assertEqual(800, result.usage.reasoning_tokens)
        self.assertAlmostEqual(0.858, result.usage.estimated_cost_usd)

    def test_openai_provider_preserves_empty_length_response_and_usage_for_batch_split(self):
        config = Config(
            moodle_base_url="https://moodle.example.com", moodle_token="token",
            moodle_service="service", openai_api_key="key", openai_model="gpt-5.6-terra",
            timezone="Asia/Jerusalem", active_course_ids=set(),
        )
        response = Mock(status_code=200, headers={"x-request-id": "req_empty"})
        response.json.return_value = {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 4000,
                "completion_tokens_details": {"reasoning_tokens": 4000},
            },
        }
        provider = OpenAIProvider(config)
        provider.session.post = Mock(return_value=response)

        result = provider.generate(
            model="gpt-5.6-terra", system_content="system",
            user_content=[{"type": "text", "text": "question"}],
            temperature=0.2, reasoning_effort="medium", max_tokens=4000,
        )

        self.assertEqual("", result.content)
        self.assertEqual("length", result.finish_reason)
        self.assertEqual(4000, result.usage.reasoning_tokens)
        self.assertEqual(1, provider.usage_totals.request_count)

    def test_llm_generator_falls_back_to_global_model(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="global-model",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        provider = Mock()
        provider.generate.return_value = "<p>answer</p>"
        registry = Mock()
        registry.get.return_value = provider
        generator = LLMAnswerGenerator(config, registry)

        result = generator.generate_specialist_output(
            self.make_assignment(),
            [],
            [],
            "solve",
            "teacher",
            "Solve carefully",
            provider_name="openai",
            model=None,
            temperature=0.3,
        )

        self.assertEqual("<p>answer</p>", result)
        self.assertEqual("global-model", provider.generate.call_args.kwargs["model"])
        self.assertEqual(0.3, provider.generate.call_args.kwargs["temperature"])

    def test_specialist_document_batches_and_preserves_inventory_order(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="model",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        questions = [
            {"id": f"2{letter}", "label": f"({letter})", "top_level": "2", "page_indices": [], "prompt_summary": "Part"}
            for letter in "abcde"
        ]
        provider = Mock()
        provider.generate.side_effect = [
            LLMResponse(json.dumps({"questions": questions}), "stop"),
            LLMResponse(json.dumps({"answers": [{"id": f"2{x}", "html": f"<p>{x}</p>"} for x in "abcd"]}), "stop"),
            LLMResponse(json.dumps({"answers": [{"id": "2e", "html": "<p>e</p>"}]}), "stop"),
        ]
        registry = Mock()
        registry.get.return_value = provider
        generator = LLMAnswerGenerator(config, registry)

        document = generator.generate_specialist_document(
            self.make_assignment(), [], [], "solve", "teacher", "Solve all parts"
        )

        self.assertTrue(document.complete)
        self.assertEqual(["2a", "2b", "2c", "2d", "2e"], [answer.id for answer in document.answers])
        self.assertLess(document.html.index("data-question-id=\"2a\""), document.html.index("data-question-id=\"2e\""))
        self.assertEqual(3, provider.generate.call_count)

    def test_specialist_document_splits_batch_when_response_is_truncated(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="model",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        inventory = {
            "questions": [
                {"id": "1a", "label": "(a)", "top_level": "1", "page_indices": [], "prompt_summary": "A"},
                {"id": "1b", "label": "(b)", "top_level": "1", "page_indices": [], "prompt_summary": "B"},
            ]
        }
        provider = Mock()
        provider.generate.side_effect = [
            LLMResponse(json.dumps(inventory), "stop"),
            LLMResponse("{}", "length"),
            LLMResponse('{"answers":[{"id":"1a","html":"<p>A</p>"}]}', "stop"),
            LLMResponse('{"answers":[{"id":"1b","html":"<p>B</p>"}]}', "stop"),
        ]
        registry = Mock()
        registry.get.return_value = provider

        document = LLMAnswerGenerator(config, registry).generate_specialist_document(
            self.make_assignment(), [], [], "solve", "teacher", "Solve"
        )

        self.assertTrue(document.complete)
        self.assertIn("length", document.finish_reasons)
        self.assertEqual({"1a": 2, "1b": 2}, document.attempts)

    def test_specialist_document_marks_missing_after_two_retries(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="model",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        inventory = {
            "questions": [
                {"id": "7f", "label": "(f)", "top_level": "7", "page_indices": [], "prompt_summary": "F"}
            ]
        }
        provider = Mock()
        provider.generate.side_effect = [
            LLMResponse(json.dumps(inventory), "stop"),
            LLMResponse('{"answers":[]}', "stop"),
            LLMResponse('{"answers":[]}', "stop"),
            LLMResponse('{"answers":[]}', "stop"),
        ]
        registry = Mock()
        registry.get.return_value = provider

        document = LLMAnswerGenerator(config, registry).generate_specialist_document(
            self.make_assignment(), [], [], "solve", "teacher", "Solve"
        )

        self.assertFalse(document.complete)
        self.assertEqual(["7f"], document.missing_ids)
        self.assertEqual(3, document.attempts["7f"])
        self.assertIn("לא הושלם", document.html)

    def test_specialist_document_renders_iupac_and_smiles_ltr(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="model",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        provider = Mock()
        provider.generate.side_effect = [
            LLMResponse('{"questions":[{"id":"1","label":"1","top_level":"1","page_indices":[],"prompt_summary":"Product"}]}', "stop"),
            LLMResponse(
                '{"answers":[{"id":"1","html":"<p>תשובה</p>","compounds":['
                '{"iupac_name":"2-methylpropene","smiles":"CC(=C)C","role":"product"}]}]}',
                "stop",
            ),
        ]
        registry = Mock()
        registry.get.return_value = provider

        document = LLMAnswerGenerator(config, registry).generate_specialist_document(
            self.make_assignment(), [], [], "solve", "teacher", "Use English IUPAC names"
        )

        self.assertIn('class="compound-references" dir="ltr"', document.html)
        self.assertIn("2-methylpropene", document.html)
        self.assertIn("CC(=C)C", document.html)

    def test_provider_registry_rejects_unsupported_provider(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="model",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )

        with self.assertRaisesRegex(WorkflowError, "Unsupported LLM provider"):
            LLMProviderRegistry(config).get("anthropic")

    def test_run_workflow_uses_bounded_subject_agent_pool(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="model",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        assignments = [
            self.make_assignment(assignment_id=1, title="Math 1"),
            self.make_assignment(assignment_id=2, title="Math 2"),
        ]
        moodle = Mock()
        moodle.get_assignments.return_value = assignments
        barrier = threading.Barrier(2)
        worker_names = []

        def fake_process(args, loaded_config, reference_date, assignment):
            worker_names.append(threading.current_thread().name)
            barrier.wait(timeout=3)
            return {"classified_topic": "mathematics", "assignment_id": assignment.assignment_id}

        args = SimpleNamespace(
            env_file=".env",
            bootstrap_token=False,
            save_token=False,
            workflow_mode="instructions",
            submit=False,
            date="2026-06-18",
            selection_window="today",
            include_submitted=False,
            max_assignments=None,
            max_workers=2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(workflow_module, "load_env_file"),
                patch.object(workflow_module, "load_config", return_value=config),
                patch.object(workflow_module, "MoodleClient", return_value=moodle),
                patch.object(workflow_module, "ensure_output_dir", return_value=Path(temp_dir)),
                patch.object(workflow_module, "process_assignment", side_effect=fake_process),
            ):
                summary = workflow_module.run_workflow(args)

        self.assertEqual(2, summary["worker_count"])
        self.assertEqual({"mathematics": 2}, summary["topic_counts"])
        self.assertEqual([1, 2], [result["assignment_id"] for result in summary["results"]])
        self.assertEqual(2, len(set(worker_names)))
        self.assertTrue(all(name.startswith("subject-agent") for name in worker_names))

    def test_process_assignment_classifies_then_dispatches_to_specialist(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="model",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        args = SimpleNamespace(
            workflow_mode="solve",
            solver_stage=False,
            submit=False,
        )
        assignment = self.make_assignment(title="POSIX Threads")
        attachment_context = AttachmentContext(
            texts=["pthread_create and mutex exercise"],
            image_urls=["data:image/png;base64,abc"],
            saved_files=[],
            preview_image_paths=[],
        )
        classifier = Mock()
        classifier.classify.return_value = TopicClassification(
            "systems_programming", 0.96, "POSIX API", "llm"
        )
        agent = Mock()
        agent.display_name = "systems programming teacher"
        agent.provider = "openai"
        agent.model = "agent-model"
        agent.temperature = 0.4
        agent.generate.return_value = GeneratedDocument(
            html="<section><p>solution</p></section>",
            inventory=[QuestionItem("1", "1", "1", (1,), "Solve")],
            answers=[QuestionAnswer("1", "<p>solution</p>")],
            missing_ids=[],
            attempts={"1": 1},
            finish_reasons=["stop"],
        )
        agent.generate_from_manifest.return_value = agent.generate.return_value
        manifest = DocumentManifest(
            schema_version=1,
            source_hash="hash",
            model="model",
            assignment_id=assignment.assignment_id,
            suggested_topic="systems_programming",
            questions=[ManifestQuestion("1", "1", "1", (1,), "pthread", "pthread", VisualEvidence())],
            created_at="now",
        )
        generator = Mock()
        generator.load_or_build_manifest.return_value = (manifest, Path("document_manifest.json"))
        generator.usage_totals.return_value = UsageTotals()

        with tempfile.TemporaryDirectory() as temp_dir:
            assignment_dir = Path(temp_dir)
            text_path = assignment_dir / "answer.txt"
            html_path = assignment_dir / "answer.html"
            metadata_path = assignment_dir / "answer.json"
            pdf_path = assignment_dir / "submission.pdf"
            text_path.write_text("solution", encoding="utf-8")
            pdf_path.write_bytes(b"%PDF")

            with (
                patch.object(workflow_module, "MoodleClient"),
                patch.object(workflow_module, "LLMAnswerGenerator", return_value=generator),
                patch.object(workflow_module, "OpenAITopicClassifier", return_value=classifier),
                patch.object(workflow_module, "get_subject_agent", return_value=agent) as agent_lookup,
                patch.object(workflow_module, "build_assignment_output_dir", return_value=assignment_dir),
                patch.object(workflow_module, "collect_attachment_context", return_value=attachment_context),
                patch.object(
                    workflow_module,
                    "write_document_bundle",
                    return_value=(text_path, html_path, metadata_path, pdf_path, "playwright"),
                ),
            ):
                result = workflow_module.process_assignment(args, config, self.target_date, assignment)

        classifier.classify.assert_not_called()
        agent_lookup.assert_called_once()
        self.assertEqual("systems_programming", agent_lookup.call_args.args[0])
        agent.generate_from_manifest.assert_called_once()
        self.assertEqual("systems_programming", result["classified_topic"])
        self.assertEqual("systems programming teacher", result["subject_agent"])
        self.assertEqual("openai", result["llm_provider"])
        self.assertEqual("agent-model", result["llm_model"])
        self.assertEqual(0.4, result["llm_temperature"])
        self.assertEqual("completed", result["status"])

    def test_process_assignment_blocks_submit_when_coverage_is_incomplete(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="model",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        args = SimpleNamespace(workflow_mode="solve", solver_stage=False, submit=True)
        assignment = self.make_assignment(title="Incomplete assignment")
        context = AttachmentContext([], [], [], [])
        classifier = Mock()
        classifier.classify.return_value = TopicClassification("general", 0.8, "General", "llm")
        agent = Mock(
            display_name="general academic teacher",
            provider="openai",
            model="model",
            temperature=0.2,
        )
        agent.generate.return_value = GeneratedDocument(
            html='<section><p class="missing-answer">Missing</p></section>',
            inventory=[QuestionItem("8", "8", "8", (1,), "Question 8")],
            answers=[],
            missing_ids=["8"],
            attempts={"8": 3},
            finish_reasons=["length", "stop", "stop"],
        )
        agent.generate_from_manifest.return_value = agent.generate.return_value
        manifest = DocumentManifest(
            schema_version=1,
            source_hash="hash",
            model="model",
            assignment_id=assignment.assignment_id,
            suggested_topic="general",
            questions=[ManifestQuestion("8", "8", "8", (1,), "Question 8", "Question 8", VisualEvidence())],
            created_at="now",
        )
        generator = Mock()
        generator.load_or_build_manifest.return_value = (manifest, Path("document_manifest.json"))
        generator.usage_totals.return_value = UsageTotals()
        moodle = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            assignment_dir = Path(temp_dir)
            text_path = assignment_dir / "answer.txt"
            html_path = assignment_dir / "answer.html"
            metadata_path = assignment_dir / "answer.json"
            pdf_path = assignment_dir / "submission.pdf"
            text_path.write_text("Missing", encoding="utf-8")
            pdf_path.write_bytes(b"%PDF")
            with (
                patch.object(workflow_module, "MoodleClient", return_value=moodle),
                patch.object(workflow_module, "LLMAnswerGenerator", return_value=generator),
                patch.object(workflow_module, "OpenAITopicClassifier", return_value=classifier),
                patch.object(workflow_module, "get_subject_agent", return_value=agent),
                patch.object(workflow_module, "build_assignment_output_dir", return_value=assignment_dir),
                patch.object(workflow_module, "collect_attachment_context", return_value=context),
                patch.object(
                    workflow_module,
                    "write_document_bundle",
                    return_value=(text_path, html_path, metadata_path, pdf_path, "playwright"),
                ),
            ):
                result = workflow_module.process_assignment(args, config, self.target_date, assignment)

        self.assertEqual("incomplete", result["status"])
        self.assertEqual(["8"], result["coverage_missing_ids"])
        self.assertIn("Missing question IDs: 8", result["submission_blocked_reason"])
        moodle.submit_assignment.assert_not_called()

    def test_openai_http_retries_rate_limit_then_returns_success(self):
        rate_limited = Mock(status_code=429, headers={"Retry-After": "0", "x-request-id": "req_rate"})
        rate_limited.json.return_value = {
            "error": {"message": "Too many requests", "type": "rate_limit_error", "code": "rate_limit_exceeded"}
        }
        rate_limited.text = "Too many requests"
        success = Mock(status_code=200, headers={})
        session = Mock()
        session.post.side_effect = [rate_limited, success]
        sleeps = []

        with patch.dict(os.environ, {"OPENAI_MIN_REQUEST_INTERVAL_SECONDS": "0"}):
            response = post_openai_json(
                session,
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": "Bearer test"},
                payload={"model": "test"},
                sleep_fn=sleeps.append,
            )

        self.assertIs(success, response)
        self.assertEqual(2, session.post.call_count)
        self.assertEqual([0.0], sleeps)

    def test_openai_http_does_not_retry_insufficient_quota(self):
        quota_error = Mock(status_code=429, headers={"x-request-id": "req_quota"})
        quota_error.json.return_value = {
            "error": {
                "message": "You exceeded your current quota.",
                "type": "insufficient_quota",
                "code": "insufficient_quota",
            }
        }
        quota_error.text = "quota exceeded"
        session = Mock()
        session.post.return_value = quota_error

        with patch.dict(os.environ, {"OPENAI_MIN_REQUEST_INTERVAL_SECONDS": "0"}):
            with self.assertRaises(OpenAIRequestError) as raised:
                post_openai_json(
                    session,
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": "Bearer test"},
                    payload={"model": "test"},
                    sleep_fn=Mock(),
                )

        self.assertEqual(1, session.post.call_count)
        self.assertEqual(429, raised.exception.status_code)
        self.assertEqual("insufficient_quota", raised.exception.error_code)
        self.assertEqual("req_quota", raised.exception.request_id)
        self.assertIn("billing", str(raised.exception))

    def test_process_assignment_safely_reports_failure_stage_and_openai_details(self):
        assignment = self.make_assignment()
        args = SimpleNamespace()
        config = Mock()
        cause = OpenAIRequestError(
            "rate limited",
            status_code=429,
            error_type="rate_limit_error",
            error_code="rate_limit_exceeded",
            request_id="req_stage",
        )

        with patch.object(
            workflow_module,
            "process_assignment",
            side_effect=WorkflowStageError("generation", cause),
        ):
            result = workflow_module.process_assignment_safely(
                args, config, self.target_date, assignment
            )

        self.assertEqual("error", result["status"])
        self.assertEqual("generation", result["failed_stage"])
        self.assertEqual(429, result["openai_status_code"])
        self.assertEqual("req_stage", result["openai_request_id"])

    def test_manifest_sends_four_pages_once_and_reuses_cache(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="gpt-4o-mini",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        questions = [
            {
                "id": str(index),
                "label": str(index),
                "top_level": str(index),
                "page_indices": [index],
                "prompt_text": f"Question {index}",
                "prompt_summary": f"Q{index}",
                "visual": {
                    "description": "diagram",
                    "bounding_box": [0.1, 0.1, 0.9, 0.9],
                    "confidence": 0.95,
                    "unreadable": False,
                },
            }
            for index in range(1, 5)
        ]
        provider = Mock()
        provider.generate.return_value = LLMResponse(
            json.dumps({
                "suggested_topic": "organic_chemistry",
                "page_audit": [
                    {"page_index": index, "fully_processed": True,
                     "visible_top_levels": [str(index)], "question_ids": [str(index)]}
                    for index in range(1, 5)
                ],
                "questions": questions,
            }),
            "stop",
        )
        registry = Mock()
        registry.get.return_value = provider
        generator = LLMAnswerGenerator(config, registry)
        context = AttachmentContext(
            texts=["short extracted text"],
            image_urls=[f"data:image/png;base64,page{index}" for index in range(1, 5)],
            saved_files=[],
            preview_image_paths=[],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            assignment_dir = Path(temp_dir)
            manifest, path = generator.load_or_build_manifest(
                self.make_assignment(), context, assignment_dir
            )
            cached, cached_path = generator.load_or_build_manifest(
                self.make_assignment(), context, assignment_dir
            )

        self.assertEqual(4, len(manifest.questions))
        self.assertEqual(1, manifest.vision_request_count)
        self.assertFalse(manifest.from_cache)
        self.assertTrue(cached.from_cache)
        self.assertEqual(path, cached_path)
        self.assertEqual(1, provider.generate.call_count)
        content = provider.generate.call_args.kwargs["user_content"]
        image_items = [item for item in content if item.get("type") == "image_url"]
        self.assertEqual(4, len(image_items))
        self.assertEqual(4, len({item["image_url"]["url"] for item in image_items}))
        response_format = provider.generate.call_args.kwargs["response_format"]
        self.assertEqual("json_schema", response_format["type"])
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(MANIFEST_SCHEMA_VERSION, manifest.schema_version)

    def test_manifest_normalizes_equivalent_identifiers_without_retry(self):
        config = Config(
            moodle_base_url="https://moodle.example.com", moodle_token="token",
            moodle_service="service", openai_api_key="key", openai_model="gpt-5.6-terra",
            timezone="Asia/Jerusalem", active_course_ids=set(),
        )
        provider = Mock()
        provider.generate.return_value = LLMResponse(json.dumps({
            "suggested_topic": "organic_chemistry",
            "page_audit": [{
                "page_index": 1, "fully_processed": True,
                "visible_top_levels": ["6"], "question_ids": ["6-a"],
            }],
            "questions": [{
                "id": "6(a)", "label": "(a)", "top_level": "6", "page_indices": [1],
                "prompt_text": "Predict the product", "prompt_summary": "Product",
                "visual": {"description": "reaction", "confidence": 1.0, "unreadable": False},
            }],
        }), "stop")
        registry = Mock()
        registry.get.return_value = provider
        context = AttachmentContext([], ["page-1"], [], [])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, _ = LLMAnswerGenerator(config, registry).load_or_build_manifest(
                self.make_assignment(), context, root
            )
            self.assertTrue((root / "document_manifest.raw.json").is_file())
            self.assertTrue((root / "manifest_validation.json").is_file())

        self.assertEqual(["6a"], [question.id for question in manifest.questions])
        self.assertEqual(0, manifest.manifest_retry_count)
        self.assertEqual(1, provider.generate.call_count)

    def test_manifest_retries_only_missing_page_and_merges_result(self):
        config = Config(
            moodle_base_url="https://moodle.example.com", moodle_token="token",
            moodle_service="service", openai_api_key="key", openai_model="gpt-5.6-terra",
            timezone="Asia/Jerusalem", active_course_ids=set(),
        )

        def question(item_id, page):
            return {
                "id": item_id, "label": item_id, "top_level": item_id,
                "page_indices": [page], "prompt_text": f"Question {item_id}",
                "prompt_summary": f"Q{item_id}",
                "visual": {"description": "", "confidence": 1.0, "unreadable": False},
            }

        provider = Mock()
        provider.generate.side_effect = [
            LLMResponse(json.dumps({
                "suggested_topic": "organic_chemistry",
                "page_audit": [{"page_index": 1, "fully_processed": True,
                                "visible_top_levels": ["1"], "question_ids": ["1"]}],
                "questions": [question("1", 1)],
            }), "stop"),
            LLMResponse(json.dumps({
                "suggested_topic": "organic_chemistry",
                "page_audit": [{"page_index": 2, "fully_processed": True,
                                "visible_top_levels": ["2"], "question_ids": ["2"]}],
                "questions": [question("2", 2)],
            }), "stop"),
        ]
        registry = Mock()
        registry.get.return_value = provider
        context = AttachmentContext([], ["page-1", "page-2"], [], [])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, _ = LLMAnswerGenerator(config, registry).load_or_build_manifest(
                self.make_assignment(), context, root
            )
            retry_raw = json.loads((root / "document_manifest_retry.raw.json").read_text("utf-8"))
            validation = json.loads((root / "manifest_validation.json").read_text("utf-8"))

        retry_content = provider.generate.call_args_list[1].kwargs["user_content"]
        retry_images = [item["image_url"]["url"] for item in retry_content if item["type"] == "image_url"]
        self.assertEqual(["page-2"], retry_images)
        self.assertEqual(["1", "2"], [item.id for item in manifest.questions])
        self.assertEqual(1, manifest.manifest_retry_count)
        self.assertEqual((2,), manifest.retried_pages)
        self.assertEqual([2], retry_raw["attempts"][0]["pages"])
        self.assertTrue(validation["chunks"][0]["final"]["valid"])

    def test_manifest_accepts_continuation_page_without_new_top_level_heading(self):
        config = Config(
            moodle_base_url="https://moodle.example.com", moodle_token="token",
            moodle_service="service", openai_api_key="key", openai_model="gpt-5.6-terra",
            timezone="Asia/Jerusalem", active_course_ids=set(),
        )
        provider = Mock()
        provider.generate.return_value = LLMResponse(json.dumps({
            "suggested_topic": "computational_models",
            "page_audit": [
                {"page_index": 1, "fully_processed": True,
                 "visible_top_levels": ["2"], "question_ids": ["2"]},
                {"page_index": 2, "fully_processed": True,
                 "visible_top_levels": [], "question_ids": ["2"]},
            ],
            "questions": [{
                "id": "2", "label": "2", "top_level": "2", "page_indices": [1, 2],
                "prompt_text": "A proof continued on the next page",
                "prompt_summary": "Prove the reduction",
                "visual": {"description": "", "confidence": 1.0, "unreadable": False},
            }],
        }), "stop")
        registry = Mock()
        registry.get.return_value = provider

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest, _ = LLMAnswerGenerator(config, registry).load_or_build_manifest(
                self.make_assignment(), AttachmentContext([], ["p1", "p2"], [], []), Path(temp_dir)
            )

        self.assertEqual(1, provider.generate.call_count)
        self.assertEqual((1, 2), manifest.questions[0].page_indices)

    def test_manifest_solution_is_text_only_and_resumes_checkpoint(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="gpt-4o-mini",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        questions = [
            ManifestQuestion(
                f"2{letter}", f"({letter})", "2", (1,), f"Prompt {letter}", f"Part {letter}", VisualEvidence()
            )
            for letter in "abcde"
        ]
        manifest = DocumentManifest(
            schema_version=2,
            source_hash="source",
            model="gpt-4o-mini",
            assignment_id=self.make_assignment().assignment_id,
            suggested_topic="organic_chemistry",
            questions=questions,
            created_at="now",
        )
        provider = Mock()
        provider.generate.side_effect = [
            LLMResponse(
                json.dumps({"answers": [{"id": f"2{x}", "html": f"<p>{x}</p>"} for x in "abcd"]}),
                "stop",
            ),
            LLMResponse('{"answers":[{"id":"2-e","html":"<p>e</p>"}]}', "stop"),
        ]
        registry = Mock()
        registry.get.return_value = provider
        generator = LLMAnswerGenerator(config, registry)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "document_manifest.json"
            checkpoint_path = root / "generation_checkpoint.json"
            document = generator.generate_specialist_document_from_manifest(
                self.make_assignment(), manifest, "solve", "teacher", "Solve",
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                page_image_paths=[],
            )
            resumed = generator.generate_specialist_document_from_manifest(
                self.make_assignment(), manifest, "solve", "teacher", "Solve",
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                page_image_paths=[],
            )

        self.assertTrue(document.complete)
        self.assertTrue(resumed.complete)
        self.assertTrue(resumed.resumed_from_checkpoint)
        self.assertEqual(2, provider.generate.call_count)
        for call in provider.generate.call_args_list:
            self.assertEqual(["text"], [item["type"] for item in call.kwargs["user_content"]])
        final_batch_prompt = provider.generate.call_args_list[1].kwargs["user_content"][0]["text"]
        self.assertIn('"id": "2a", "answer_requested": false', final_batch_prompt)
        self.assertIn('"id": "2e", "answer_requested": true', final_batch_prompt)

    def test_partial_manifest_resume_does_not_resend_completed_pages(self):
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="gpt-4o-mini",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )

        def manifest_response(ids):
            return LLMResponse(json.dumps({
                "suggested_topic": "general",
                "page_audit": [
                    {"page_index": value, "fully_processed": True,
                     "visible_top_levels": [str(value)], "question_ids": [str(value)]}
                    for value in ids
                ],
                "questions": [
                    {"id": str(value), "label": str(value), "top_level": str(value),
                     "page_indices": [value], "prompt_text": f"Q{value}", "prompt_summary": f"Q{value}",
                     "visual": {"confidence": 1.0, "unreadable": False}}
                    for value in ids
                ],
            }), "stop")

        provider = Mock()
        provider.generate.side_effect = [
            manifest_response(range(1, 5)),
            WorkflowError("temporary failure"),
            manifest_response(range(5, 9)),
        ]
        registry = Mock()
        registry.get.return_value = provider
        generator = LLMAnswerGenerator(config, registry)
        context = AttachmentContext(
            texts=["source"],
            image_urls=[f"page-{index}" for index in range(1, 9)],
            saved_files=[],
            preview_image_paths=[],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(WorkflowError):
                generator.load_or_build_manifest(self.make_assignment(), context, root)
            self.assertTrue((root / "document_manifest.partial.json").exists())
            manifest, _ = generator.load_or_build_manifest(self.make_assignment(), context, root)

        self.assertEqual(8, len(manifest.questions))
        self.assertEqual(3, provider.generate.call_count)
        resumed_content = provider.generate.call_args_list[2].kwargs["user_content"]
        resumed_images = [item["image_url"]["url"] for item in resumed_content if item["type"] == "image_url"]
        self.assertEqual(["page-5", "page-6", "page-7", "page-8"], resumed_images)

    def test_low_confidence_manifest_uses_one_crop_then_text_solution(self):
        if not PYMUPDF_AVAILABLE:
            self.skipTest("PyMuPDF is required for deterministic crop testing")
        config = Config(
            moodle_base_url="https://moodle.example.com",
            moodle_token="token",
            moodle_service="service",
            openai_api_key="key",
            openai_model="gpt-4o-mini",
            timezone="Asia/Jerusalem",
            active_course_ids=set(),
        )
        question = ManifestQuestion(
            "1", "1", "1", (1,), "unclear", "unclear",
            VisualEvidence(description="unclear", bounding_box=(0.0, 0.0, 1.0, 1.0), confidence=0.4, unreadable=True),
        )
        manifest = DocumentManifest(2, "source", "gpt-4o-mini", 456, "organic_chemistry", [question], "now")
        corrected = {
            "question": {
                "id": "1", "label": "1", "top_level": "1", "page_indices": [1],
                "prompt_text": "Predict the product", "prompt_summary": "Product",
                "visual": {"description": "clear reaction", "bounding_box": [0, 0, 1, 1],
                           "confidence": 0.99, "unreadable": False},
            }
        }
        provider = Mock()
        provider.generate.side_effect = [
            LLMResponse(json.dumps(corrected), "stop"),
            LLMResponse('{"answers":[{"id":"1","html":"<p>The reaction proceeds by an SN2 mechanism.</p>"}]}', "stop"),
        ]
        registry = Mock()
        registry.get.return_value = provider
        generator = LLMAnswerGenerator(config, registry)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "page.pdf"
            output_module.build_answer_pdf(pdf_path, "Question", ["Reaction diagram"])
            png_bytes = render_pdf_to_png_bytes(pdf_path.read_bytes(), max_pages=1)[0]
            image_path = root / "page.png"
            image_path.write_bytes(png_bytes)
            manifest_path = root / "document_manifest.json"
            checkpoint_path = root / "generation_checkpoint.json"
            document = generator.generate_specialist_document_from_manifest(
                self.make_assignment(), manifest, "solve", "teacher", "Solve",
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                page_image_paths=[str(image_path)],
            )
            saved = load_manifest(manifest_path)

        self.assertTrue(document.complete)
        self.assertEqual(2, provider.generate.call_count)
        fallback_content = provider.generate.call_args_list[0].kwargs["user_content"]
        solution_content = provider.generate.call_args_list[1].kwargs["user_content"]
        self.assertEqual(1, sum(item["type"] == "image_url" for item in fallback_content))
        self.assertEqual(0, sum(item["type"] == "image_url" for item in solution_content))
        self.assertTrue(saved.questions[0].visual.fallback_used)
        self.assertEqual(1, saved.fallback_crop_count)

    def test_manifest_rejects_missing_page_audit_instead_of_claiming_complete(self):
        config = Config(
            moodle_base_url="https://moodle.example.com", moodle_token="token",
            moodle_service="service", openai_api_key="key", openai_model="gpt-4o-mini",
            timezone="Asia/Jerusalem", active_course_ids=set(),
        )
        provider = Mock()
        provider.generate.return_value = LLMResponse(json.dumps({
            "suggested_topic": "organic_chemistry",
            "page_audit": [
                {"page_index": 1, "fully_processed": True,
                 "visible_top_levels": ["2"], "question_ids": ["2a"]},
            ],
            "questions": [{
                "id": "2a", "label": "(a)", "top_level": "2", "page_indices": [1],
                "prompt_text": "one page only", "prompt_summary": "one page only",
                "visual": {"description": "reaction", "confidence": 1.0},
            }],
        }), "stop")
        registry = Mock()
        registry.get.return_value = provider
        context = AttachmentContext(
            texts=[], image_urls=[f"page-{index}" for index in range(1, 5)],
            saved_files=[], preview_image_paths=[],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(WorkflowError, "page"):
                LLMAnswerGenerator(config, registry).load_or_build_manifest(
                    self.make_assignment(), context, root
                )
            self.assertFalse((root / "document_manifest.json").exists())
            self.assertTrue((root / "document_manifest.raw.json").is_file())
            self.assertTrue((root / "document_manifest_retry.raw.json").is_file())
            self.assertTrue((root / "manifest_validation.json").is_file())
        self.assertEqual(2, provider.generate.call_count)

    def test_placeholder_answer_is_not_substantive(self):
        self.assertFalse(is_substantive_answer_html("<p>...</p>"))
        self.assertFalse(is_substantive_answer_html("<p>…</p>"))
        self.assertTrue(is_substantive_answer_html("<p>Use sodium ethoxide because it favors Zaitsev elimination.</p>"))

    def test_normalize_html_unwraps_math_code_and_replaces_unsupported_textsc(self):
        normalized = normalize_html_fragment(
            r"<p><code>\(NP=coNP\)</code> and <code>literal-code</code> "
            r"\(\textsc{S-T-Con}\)</p>"
        )

        self.assertIn(r"\(NP=coNP\)", normalized)
        self.assertNotIn(r"<code>\(NP=coNP\)</code>", normalized)
        self.assertIn("<code>literal-code</code>", normalized)
        self.assertIn(r"\mathrm{S-T-Con}", normalized)
        self.assertNotIn(r"\textsc", normalized)

    def test_ipad_html_hides_assistive_mathml_that_can_expand_print_width(self):
        document = output_module.build_rich_html_document(
            "בדיקת נוסחה",
            r"<p>נוסחה: \(L \in NP\)</p>",
            output_style="ipad_clean",
            paper_style="grid",
            handwriting_font="personal",
        )

        self.assertIn("mjx-assistive-mml { display: none !important; }", document)
        self.assertIn("enableAssistiveMml: false", document)
        self.assertIn("querySelectorAll('mjx-assistive-mml')", document)
        self.assertIn("break-after: avoid-page", document)

    def test_normalize_html_wraps_bare_tex_inside_ltr_spans(self):
        normalized = normalize_html_fragment(
            r'<p><span dir="ltr">A\in NP</span> '
            r'<span dir="ltr">2\text{-}Coloring\)</span> '
            r'<span dir="ltr">\(P=NP\)</span> '
            r'<span dir="ltr">plain LTR</span></p>'
        )

        self.assertIn(r'<span dir="ltr">\(A\in NP\)</span>', normalized)
        self.assertIn(r'<span dir="ltr">\(2\text{-}Coloring\)</span>', normalized)
        self.assertIn(r'<span dir="ltr">\(P=NP\)</span>', normalized)
        self.assertIn(r'<span dir="ltr">plain LTR</span>', normalized)

    def test_normalize_html_repairs_orphan_math_closer_in_unclosed_span(self):
        normalized = normalize_html_fragment(
            r'<p>אל <span dir="ltr">2\text{-}Coloring\), אפשר להכריע '
            r'<span dir="ltr">\(A\in NP\)</span></p>'
        )

        self.assertIn(r'<span dir="ltr">\(2\text{-}Coloring\), אפשר', normalized)
        self.assertIn(r'<span dir="ltr">\(A\in NP\)</span>', normalized)

    def test_canonical_top_level_extracts_question_number_from_descriptive_heading(self):
        self.assertEqual("1", canonical_top_level("Question 1 - Complexity Classes"))
        self.assertEqual("3", canonical_top_level("3.3"))

    def test_hebrew_answer_validation_rejects_english_and_model_leakage(self):
        self.assertFalse(is_acceptable_answer_html(
            "<p>Assume the language is in NP. Therefore the claim follows.</p>", "he"
        ))
        self.assertFalse(is_acceptable_answer_html(
            "<p>נוכיח את הטענה כנדרש.</p><p>[assistantanalysis] Need final JSON.</p>", "he"
        ))
        self.assertTrue(is_acceptable_answer_html(
            "<p>נניח שהשפה שייכת למחלקה NP, ונוכיח את שני כיווני הטענה באמצעות הפחתה פולינומית.</p>",
            "he",
        ))
        self.assertTrue(is_acceptable_answer_html(
            "<p>נבדוק כל פסוקית ונראה שקיימים בה ערך אמת וערך שקר, ולכן ההשמה המבוקשת תקינה.</p>"
            "<code>\\(c_i=(x_1 \\lor x_2 \\lor x_3), "
            "x_1=T, x_2=F, x_3=T, NAT, CNF, SAT\\)</code>",
            "he",
        ))

    def test_valid_smiles_renders_svg_and_invalid_smiles_is_flagged(self):
        svg = smiles_to_svg("CC(=C)C")
        self.assertIsNotNone(svg)
        self.assertIn("<svg", svg)
        self.assertIn("#163D72", svg)
        self.assertIn("#FFFFFF00", svg)
        rendered = compound_reference_html("2-methylpropene", "CC(=C)C", "product")
        self.assertIn("<svg", rendered)
        invalid = compound_reference_html("not a molecule", "NaOH", "product")
        self.assertNotIn("<svg", invalid)
        self.assertIn("Invalid SMILES", invalid)


if __name__ == "__main__":
    unittest.main()
