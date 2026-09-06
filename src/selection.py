import datetime as dt
from zoneinfo import ZoneInfo

from .models import Assignment, Config, SelectionWindow


def assignment_is_relevant(
    assignment: Assignment,
    selection_window: SelectionWindow,
    include_submitted: bool,
    include_closed: bool = False,
) -> bool:
    return evaluate_assignment(
        assignment, selection_window, include_submitted, include_closed=include_closed
    )["relevant"]


def evaluate_assignment(
    assignment: Assignment,
    selection_window: SelectionWindow,
    include_submitted: bool,
    now: dt.datetime | None = None,
    include_closed: bool = False,
) -> dict[str, object]:
    now = now or (
        dt.datetime.now(assignment.due_at.tzinfo) if assignment.due_at and assignment.due_at.tzinfo else None
    )

    if assignment.due_at is None:
        return {"relevant": False, "reason": "missing_due_date"}
    if assignment.due_at.date() < selection_window.start_date:
        return {"relevant": False, "reason": "due_before_window"}
    if assignment.due_at.date() > selection_window.end_date:
        return {"relevant": False, "reason": "due_after_window"}
    if assignment.allow_submissions_from and now and now < assignment.allow_submissions_from:
        return {"relevant": False, "reason": "submission_window_not_open"}
    is_closed = bool(assignment.cutoff_at and now and now > assignment.cutoff_at)
    if is_closed and not include_closed:
        return {"relevant": False, "reason": "submission_window_closed"}
    if assignment.already_submitted and not include_submitted:
        return {"relevant": False, "reason": "already_submitted"}
    if selection_window.mode == "today":
        reason = "due_today_closed_included" if is_closed else "due_today"
        return {"relevant": True, "reason": reason}
    reason = "due_by_end_of_week_closed_included" if is_closed else "due_by_end_of_week"
    return {"relevant": True, "reason": reason}


def resolve_target_date(config: Config, value: str | None) -> dt.date:
    if value:
        return dt.date.fromisoformat(value)
    return dt.datetime.now(ZoneInfo(config.timezone)).date()


def resolve_selection_window(reference_date: dt.date, mode: str) -> SelectionWindow:
    if mode == "today":
        return SelectionWindow(mode=mode, start_date=reference_date, end_date=reference_date)

    days_until_saturday = (5 - reference_date.weekday()) % 7
    end_date = reference_date + dt.timedelta(days=days_until_saturday)
    return SelectionWindow(mode=mode, start_date=reference_date, end_date=end_date)
