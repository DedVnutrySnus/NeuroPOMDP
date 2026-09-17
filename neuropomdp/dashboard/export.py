"""Export helpers for dashboard result objects."""

from __future__ import annotations

import json
from typing import Sequence

from ..types import BenchmarkResult, EpisodeResult, SweepResult, to_serializable
from .formatting import benchmark_rows, episode_rows, rows_to_csv_text, sweep_rows


def rows_to_csv_bytes(
    rows: Sequence[dict[str, object]] | None,
    fieldnames: Sequence[str] | None = None,
) -> bytes:
    """Serialize tabular rows to UTF-8 CSV bytes."""

    return rows_to_csv_text(rows or [], fieldnames).encode("utf-8")


def main_metrics_rows(result: object | None) -> list[dict[str, object]]:
    """Return the primary metric rows for a dashboard result."""

    if isinstance(result, EpisodeResult):
        return [
            {
                "Total Reward": float(result.total_reward),
                "Success": bool(result.success),
                "Mean Information Gain": float(result.mean_information_gain),
                "Mean Pragmatic Cost": float(result.mean_pragmatic_cost),
            }
        ]
    if isinstance(result, BenchmarkResult):
        return benchmark_rows(result)
    if isinstance(result, SweepResult):
        return sweep_rows(result)
    return []


def timestep_rows(result: object | None) -> list[dict[str, object]]:
    """Return timestep-level rows available in a dashboard result."""

    if isinstance(result, EpisodeResult):
        return episode_rows(result)
    if isinstance(result, BenchmarkResult):
        rows: list[dict[str, object]] = []
        for agent_name, episodes in result.episodes_by_agent.items():
            for episode_index, episode in enumerate(episodes):
                for row in episode_rows(episode):
                    rows.append({"Agent": agent_name, "Episode": episode_index, **row})
        return rows
    return []


def metrics_csv_bytes(result: object | None) -> bytes:
    """Serialize primary result metrics as CSV bytes."""

    return rows_to_csv_bytes(main_metrics_rows(result))


def timestep_csv_bytes(result: object | None) -> bytes:
    """Serialize available timestep data as CSV bytes."""

    return rows_to_csv_bytes(timestep_rows(result))


def result_json_bytes(result: object | None, config: object | None = None) -> bytes:
    """Serialize run configuration, model parameters, and result data as JSON."""

    model_parameters = {}
    if config is not None:
        environment = getattr(config, "env", None)
        model_parameters = to_serializable(getattr(environment, "model", {}))

    payload = {
        "run_config": to_serializable(config) if config is not None else {},
        "model_parameters": model_parameters,
        "results": to_serializable(result) if result is not None else {},
    }
    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8")
