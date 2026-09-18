"""Local experiment history storage and dashboard view."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import streamlit as st

from ..types import to_serializable
from .export import main_metrics_rows
from .i18n import t


_HISTORY_ROOT_NAME = "NeuroPOMDP"
_HISTORY_DIR_NAME = "history"


def get_history_dir(base_dir: str | Path | None = None) -> Path:
    """Return the local history directory without touching experiment outputs."""

    if base_dir is not None:
        root = Path(base_dir)
    else:
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    return root / _HISTORY_ROOT_NAME / _HISTORY_DIR_NAME


def save_run(
    mode: str,
    config: object,
    result: object,
    history_dir: str | Path | None = None,
) -> dict[str, object]:
    """Persist one completed run as a standalone JSON record."""

    run_id = uuid.uuid4().hex
    record: dict[str, object] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": str(mode),
        "configuration": to_serializable(config),
        "metrics": main_metrics_rows(result),
    }
    target_dir = get_history_dir(history_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{run_id}.json"
    temporary_path = target_dir / f".{run_id}.tmp"
    temporary_path.write_text(json.dumps(record, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary_path.replace(target_path)
    return record


def list_runs(history_dir: str | Path | None = None) -> list[dict[str, object]]:
    """Load valid history records, newest first."""

    directory = get_history_dir(history_dir)
    if not directory.exists():
        return []

    records: list[dict[str, object]] = []
    for path in directory.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(record, dict) and record.get("run_id"):
            records.append(record)
    return sorted(records, key=lambda item: str(item.get("timestamp", "")), reverse=True)


def load_run(run_id: str, history_dir: str | Path | None = None) -> dict[str, object] | None:
    """Load one history record by run id."""

    for record in list_runs(history_dir):
        if record.get("run_id") == run_id:
            return record
    return None


def compare_runs(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Flatten selected history records into a comparison table."""

    rows: list[dict[str, object]] = []
    for record in records:
        metrics = record.get("metrics", [])
        if not isinstance(metrics, list):
            continue
        for metric_row in metrics:
            if not isinstance(metric_row, dict):
                continue
            row: dict[str, object] = {
                "Run ID": record.get("run_id", ""),
                "Timestamp": record.get("timestamp", ""),
                "Mode": record.get("mode", ""),
            }
            row.update(metric_row)
            rows.append(row)
    return rows


def render_history_page(history_dir: str | Path | None = None) -> None:
    """Render the local run list, configuration viewer, and comparison view."""

    st.header(t("history_heading"))
    records = list_runs(history_dir)
    if not records:
        st.info(t("history_empty"))
        return

    labels = {
        str(record["run_id"]): f"{record.get('timestamp', '')} | {record.get('mode', '')}"
        for record in records
    }
    selected_id = st.selectbox(
        t("history_select_run"),
        options=list(labels),
        format_func=lambda run_id: labels[run_id],
    )
    selected = load_run(selected_id, history_dir)
    if selected is not None:
        st.subheader(t("history_configuration"))
        st.json(selected.get("configuration", {}))
        st.subheader(t("history_metrics"))
        st.json(selected.get("metrics", []))

    st.subheader(t("history_compare"))
    compare_options = [str(record["run_id"]) for record in records]
    selected_ids = st.multiselect(
        t("history_compare_select"),
        options=compare_options,
        default=compare_options[:2],
        format_func=lambda run_id: labels[run_id],
    )
    selected_records = [record for record in records if record.get("run_id") in selected_ids]
    comparison_rows = compare_runs(selected_records)
    if comparison_rows:
        st.dataframe(comparison_rows, width="stretch", hide_index=True)
    else:
        st.info(t("history_no_metrics"))
