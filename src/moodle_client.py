import datetime as dt
import textwrap
from typing import Any

import requests
from zoneinfo import ZoneInfo

from .models import Assignment, Config, WorkflowError


def timestamp_to_datetime(value: Any, timezone: ZoneInfo) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromtimestamp(int(value), tz=timezone)


def extract_submission_status(status_payload: dict[str, Any]) -> str:
    last_attempt = status_payload.get("lastattempt") or {}
    submission = last_attempt.get("submission") or {}
    return submission.get("status") or "unknown"


def request_moodle_token(base_url: str, username: str, password: str, service: str) -> str:
    response = requests.post(
        f"{base_url}/login/token.php",
        data={"username": username, "password": password, "service": service},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("token")
    if token:
        return token
    raise WorkflowError(payload.get("error", "Moodle token request failed."))


def request_moodle_token_from_env(base_url: str, service: str) -> str:
    import os

    username = os.environ.get("MOODLE_USERNAME", "").strip()
    password = os.environ.get("MOODLE_PASSWORD", "").strip()
    if not username or not password:
        return ""
    return request_moodle_token(base_url, username, password, service)


class MoodleClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()

    @property
    def rest_url(self) -> str:
        return f"{self.config.moodle_base_url}/webservice/rest/server.php"

    @property
    def upload_url(self) -> str:
        return f"{self.config.moodle_base_url}/webservice/upload.php"

    def call(self, function_name: str, **params: Any) -> Any:
        payload = {
            "wstoken": self.config.moodle_token,
            "wsfunction": function_name,
            "moodlewsrestformat": "json",
        }
        payload.update(params)

        response = self.session.post(self.rest_url, data=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("exception"):
            raise WorkflowError(
                f"Moodle API error for {function_name}: {data.get('message', 'Unknown error')}"
            )

        return data

    def get_assignments(self) -> list[Assignment]:
        timezone = ZoneInfo(self.config.timezone)
        data = self.call("mod_assign_get_assignments")
        assignments: list[Assignment] = []

        for course in data.get("courses", []):
            course_id = int(course.get("id", 0))
            if self.config.active_course_ids and course_id not in self.config.active_course_ids:
                continue

            course_name = course.get("fullname", "Unknown Course")
            for item in course.get("assignments", []):
                assignments.append(
                    Assignment(
                        course_id=course_id,
                        course_name=course_name,
                        assignment_id=int(item.get("id", 0)),
                        title=item.get("name", "Untitled Assignment"),
                        intro_html=item.get("intro", "") or "",
                        due_at=timestamp_to_datetime(item.get("duedate"), timezone),
                        allow_submissions_from=timestamp_to_datetime(
                            item.get("allowsubmissionsfromdate"),
                            timezone,
                        ),
                        cutoff_at=timestamp_to_datetime(item.get("cutoffdate"), timezone),
                        attachments=item.get("introattachments", []) or [],
                    )
                )

        return assignments

    def mark_submission_status(self, assignments: list[Assignment]) -> None:
        timezone = ZoneInfo(self.config.timezone)
        for assignment in assignments:
            try:
                status = self.call(
                    "mod_assign_get_submission_status",
                    assignid=assignment.assignment_id,
                )
            except Exception:
                assignment.already_submitted = False
                continue

            last_attempt = status.get("lastattempt") or {}
            submission = last_attempt.get("submission") or {}
            assignment.submission_status = submission.get("status")
            assignment.already_submitted = assignment.submission_status == "submitted"
            assignment.submission_last_modified = timestamp_to_datetime(
                submission.get("timemodified"),
                timezone,
            )

    def download_attachment_text(self, attachment: dict[str, Any]) -> str:
        content, _ = self.download_attachment_bytes(attachment)
        return content.decode("utf-8", errors="replace")

    def download_attachment_bytes(self, attachment: dict[str, Any]) -> tuple[bytes, str]:
        file_url = attachment.get("fileurl")
        if not file_url:
            return b"", ""

        response = self.session.get(
            file_url,
            params={"token": self.config.moodle_token},
            timeout=60,
        )
        response.raise_for_status()
        return response.content, response.headers.get("Content-Type", "")

    def upload_file(self, file_path) -> int:
        with file_path.open("rb") as handle:
            response = self.session.post(
                self.upload_url,
                data={"token": self.config.moodle_token, "filearea": "draft", "itemid": 0},
                files={"file_1": (file_path.name, handle, "application/pdf")},
                timeout=120,
            )

        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("exception"):
            raise WorkflowError(data.get("message", "Upload failed."))
        if not isinstance(data, list) or not data:
            raise WorkflowError("Upload response did not include a draft item id.")

        item_id = data[0].get("itemid")
        if item_id is None:
            raise WorkflowError("Upload response did not include itemid.")

        return int(item_id)

    def submit_assignment(self, assignment: Assignment, pdf_path, answer_text: str) -> Any:
        item_id = self.upload_file(pdf_path)
        summary = textwrap.shorten(answer_text.replace("\n", " "), width=400, placeholder="...")

        self.call(
            "mod_assign_save_submission",
            assignmentid=assignment.assignment_id,
            **{
                "plugindata[files_filemanager]": item_id,
                "plugindata[onlinetext_editor][text]": summary,
                "plugindata[onlinetext_editor][format]": 1,
                "plugindata[onlinetext_editor][itemid]": 0,
            },
        )

        status = self.call(
            "mod_assign_get_submission_status",
            assignid=assignment.assignment_id,
        )
        return {
            "draft_item_id": item_id,
            "submission_status": extract_submission_status(status),
            "raw_status": status,
        }
