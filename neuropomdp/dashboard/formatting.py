"""Pure formatting helpers for the NeuroPOMDP dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import csv
from typing import Iterable, Sequence

import numpy as np

from ..types import ActionID, BenchmarkResult, EpisodeResult, MultiSeedSweepResult, SweepResult


STATE_LABELS = [
    "Left secret",
    "Right secret",
    "Left revealed",
    "Right revealed",
    "Left success",
    "Left failure",
    "Right success",
    "Right failure",
]

OBSERVATION_LABELS = [
    "Ambiguous",
    "Left cue",
    "Right cue",
    "Success",
    "Failure",
]

ACTION_LABELS = [
    "Inspect",
    "Exploit left",
    "Exploit right",
]

AGENT_LABELS = {
    "random": "Random",
    "pragmatic_only": "Pragmatic-only",
    "active_inference": "Active Inference",
    "bayesian_utility": "Bayesian Utility",
    "no_epistemic": "Without Epistemic Value",
}


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """Diagnostics for a single timestep in one episode."""

    step_index: int
    observation: int
    hidden_state: int
    belief: np.ndarray
    posterior_entropy: float
    free_energy: float
    selected_action: int | None
    selected_action_probability: float | None
    action_probabilities: np.ndarray
    pragmatic_values: np.ndarray
    epistemic_values: np.ndarray
    efe_values: np.ndarray


def state_label(index: int) -> str:
    """Return a readable label for a hidden state."""

    if 0 <= index < len(STATE_LABELS):
        return STATE_LABELS[index]
    return f"State {index}"


def observation_label(index: int) -> str:
    """Return a readable label for an observation."""

    if 0 <= index < len(OBSERVATION_LABELS):
        return OBSERVATION_LABELS[index]
    return f"Observation {index}"


def action_label(index: int) -> str:
    """Return a readable label for an action."""

    if 0 <= index < len(ACTION_LABELS):
        return ACTION_LABELS[index]
    return f"Action {index}"


def agent_label(name: str) -> str:
    """Return a readable label for an agent identifier."""

    return AGENT_LABELS.get(name, name.replace("_", " ").title())


def format_float(value: float, digits: int = 2) -> str:
    """Format a floating-point value for display."""

    return f"{float(value):.{digits}f}"


def format_percent(value: float, digits: int = 1) -> str:
    """Format a probability or rate as a percentage."""

    return f"{float(value) * 100.0:.{digits}f}%"


def format_probability_bar(value: float, width: int = 14) -> str:
    """Render a compact ASCII bar for a probability."""

    clipped = max(0.0, min(1.0, float(value)))
    filled = int(round(clipped * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def extract_step_diagnostics(episode: EpisodeResult, step_index: int) -> StepDiagnostics:
    """Extract the diagnostics for one timestep."""

    max_step = max(0, len(episode.beliefs) - 1)
    index = max(0, min(int(step_index), max_step))
    belief = np.asarray(episode.beliefs[index], dtype=np.float64)
    observation = int(episode.observations[index])
    hidden_state = int(episode.hidden_states[index])
    posterior_entropy = float(episode.posterior_entropies[index])
    free_energies = getattr(episode, "free_energies", [float("nan")] * len(episode.beliefs))
    free_energy = float(free_energies[index])

    if index < len(episode.actions):
        selected_action = int(episode.actions[index])
        action_probabilities = np.asarray(episode.action_probabilities[index], dtype=np.float64)
        pragmatic_values = np.asarray(episode.pragmatic_values[index], dtype=np.float64)
        epistemic_values = np.asarray(episode.epistemic_values[index], dtype=np.float64)
        efe_values = np.asarray(episode.efe_values[index], dtype=np.float64)
        selected_action_probability = float(action_probabilities[selected_action])
    else:
        num_actions = len(episode.action_probabilities[0]) if episode.action_probabilities else 0
        selected_action = None
        selected_action_probability = None
        action_probabilities = np.full((num_actions,), 1.0 / float(num_actions)) if num_actions else np.asarray([])
        pragmatic_values = np.asarray([])
        epistemic_values = np.asarray([])
        efe_values = np.asarray([])

    return StepDiagnostics(
        step_index=index,
        observation=observation,
        hidden_state=hidden_state,
        belief=belief,
        posterior_entropy=posterior_entropy,
        free_energy=free_energy,
        selected_action=selected_action,
        selected_action_probability=selected_action_probability,
        action_probabilities=action_probabilities,
        pragmatic_values=pragmatic_values,
        epistemic_values=epistemic_values,
        efe_values=efe_values,
    )


def step_action_rows(step: StepDiagnostics) -> list[dict[str, object]]:
    """Build a display row for each available action."""

    rows: list[dict[str, object]] = []
    for index, action_name in enumerate(ACTION_LABELS[: len(step.efe_values)]):
        row = {
            "Action": action_name,
            "Pragmatic": float(step.pragmatic_values[index]),
            "Epistemic": float(step.epistemic_values[index]),
            "Total EFE": float(step.efe_values[index]),
            "Policy Probability": float(step.action_probabilities[index]),
        }
        if step.selected_action is not None and index == step.selected_action:
            row["Selected"] = True
        rows.append(row)
    return rows


def generate_action_explanation(step: StepDiagnostics) -> str:
    """Generate a short explanation of the selected action."""

    if step.selected_action is None or step.efe_values.size == 0:
        return "No action was selected for this timestep."

    action_names = [action_label(index) for index in range(len(step.efe_values))]
    selected = int(step.selected_action)
    selected_name = action_names[selected]
    selected_pragmatic = float(step.pragmatic_values[selected])
    selected_epistemic = float(step.epistemic_values[selected])
    selected_total = float(step.efe_values[selected])
    selected_probability = float(step.selected_action_probability or 0.0)

    best_index = int(np.argmin(step.efe_values))
    best_name = action_names[best_index]
    best_total = float(step.efe_values[best_index])

    pragmatic_rank = int(np.argsort(step.pragmatic_values).tolist().index(selected))
    epistemic_rank = int(np.argsort(-step.epistemic_values).tolist().index(selected))
    entropy = float(step.posterior_entropy)

    parts: list[str] = []
    if selected == best_index:
        parts.append(
            f"{selected_name} had the lowest total EFE ({format_float(selected_total)})."
        )
    else:
        parts.append(
            f"{selected_name} was sampled with probability {format_percent(selected_probability)} "
            f"even though {best_name} had the lower total EFE ({format_float(best_total)})."
        )

    if selected_epistemic >= float(np.max(step.epistemic_values)) - 1e-6:
        parts.append(
            f"It provides the strongest expected information gain ({format_float(selected_epistemic)})."
        )
    elif epistemic_rank <= 1:
        parts.append(
            f"Its epistemic value is among the strongest in the action set ({format_float(selected_epistemic)})."
        )

    if selected_pragmatic <= float(np.min(step.pragmatic_values)) + 1e-6:
        parts.append(
            f"It also keeps pragmatic cost low ({format_float(selected_pragmatic)})."
        )
    elif pragmatic_rank <= 1:
        parts.append(
            f"Its pragmatic cost is competitive ({format_float(selected_pragmatic)})."
        )

    if entropy > 0.5 and selected_epistemic >= float(np.mean(step.epistemic_values)):
        parts.append(
            "The posterior is still uncertain, so epistemic value matters in the decision."
        )
    elif entropy <= 0.5:
        parts.append("Belief uncertainty is already modest, so the policy leans more on value.")

    return " ".join(parts)


def benchmark_rows(result: BenchmarkResult, agent_names: Sequence[str] | None = None) -> list[dict[str, object]]:
    """Convert benchmark summaries into tabular rows."""

    summary_map = {summary.agent_name: summary for summary in result.summaries}
    ordered_names = list(agent_names) if agent_names is not None else list(result.agent_names)
    rows: list[dict[str, object]] = []
    for name in ordered_names:
        summary = summary_map.get(name)
        if summary is None:
            continue
        rows.append(
            {
                "Agent": agent_label(name),
                "Mean Reward": float(summary.mean_reward),
                "Reward CI 95% Low": float(summary.reward_ci95_low),
                "Reward CI 95% High": float(summary.reward_ci95_high),
                "Success Rate": float(summary.success_rate),
                "Success CI 95% Low": float(summary.success_ci95_low),
                "Success CI 95% High": float(summary.success_ci95_high),
                "Mean Entropy": float(summary.mean_posterior_entropy),
                "Mean Information Gain": float(summary.mean_information_gain),
                "Info-Seeking Frequency": float(summary.info_action_frequency),
                "Inspect Count": int(summary.action_counts.get("inspect", 0)),
                "Exploit Left Count": int(summary.action_counts.get("exploit_left", 0)),
                "Exploit Right Count": int(summary.action_counts.get("exploit_right", 0)),
            }
        )
    return rows


def benchmark_delta(result: BenchmarkResult, base_agent: str, compare_agent: str) -> dict[str, float]:
    """Compute the difference between two benchmark summaries."""

    summary_map = {summary.agent_name: summary for summary in result.summaries}
    base = summary_map[base_agent]
    compare = summary_map[compare_agent]
    return {
        "reward": float(compare.mean_reward - base.mean_reward),
        "success_rate": float(compare.success_rate - base.success_rate),
        "entropy": float(compare.mean_posterior_entropy - base.mean_posterior_entropy),
        "information_gain": float(compare.mean_information_gain - base.mean_information_gain),
        "info_frequency": float(compare.info_action_frequency - base.info_action_frequency),
    }


def sweep_rows(result: SweepResult | MultiSeedSweepResult) -> list[dict[str, object]]:
    """Convert a sweep result into long-form tabular rows."""

    rows: list[dict[str, object]] = []
    seed_rewards = None
    if isinstance(result, MultiSeedSweepResult) and result.per_seed_results:
        seed_rewards = np.asarray(
            [seed_result.mean_reward for seed_result in result.per_seed_results],
            dtype=np.float64,
        )
    for noise_index, noise in enumerate(result.noise_values):
        for agent_index, agent_name in enumerate(result.agent_names):
            row: dict[str, object] = {
                "Observation Noise": float(noise),
                "Agent": agent_label(agent_name),
                "Mean Reward": float(result.mean_reward[noise_index][agent_index]),
                "Success Rate": float(result.success_rate[noise_index][agent_index]),
                "Mean Entropy": float(result.mean_entropy[noise_index][agent_index]),
                "Info-Seeking Frequency": float(result.info_action_frequency[noise_index][agent_index]),
            }
            if seed_rewards is not None and seed_rewards.shape[0] > 1:
                values = seed_rewards[:, noise_index, agent_index]
                margin = 1.96 * float(np.std(values, ddof=0)) / np.sqrt(seed_rewards.shape[0])
                row["Reward CI 95% Low"] = float(result.mean_reward[noise_index][agent_index] - margin)
                row["Reward CI 95% High"] = float(result.mean_reward[noise_index][agent_index] + margin)
            rows.append(row)
    return rows


def sweep_effect_rows(result: MultiSeedSweepResult) -> list[dict[str, object]]:
    """Convert a multi-seed sweep effect summary into research table rows."""

    from ..experiments import summarize_sweep_effect

    effect = summarize_sweep_effect(result.per_seed_results)
    rows: list[dict[str, object]] = []
    for index, noise in enumerate(effect["noise_values"]):
        reward_delta = float(effect["reward_deltas"][index])
        success_delta = float(effect["success_deltas"][index])
        reward_low = float(effect["reward_delta_ci95_low"][index])
        reward_high = float(effect["reward_delta_ci95_high"][index])
        success_low = float(effect["success_delta_ci95_low"][index])
        success_high = float(effect["success_delta_ci95_high"][index])
        rows.append(
            {
                "Observation Noise": float(noise),
                "Reward Delta": reward_delta,
                "Reward Delta CI 95% Low": reward_low,
                "Reward Delta CI 95% High": reward_high,
                "Success Delta": success_delta,
                "Success Delta CI 95% Low": success_low,
                "Success Delta CI 95% High": success_high,
                "Reliable Reward Advantage": reward_low > 0.0,
                "Reliable Success Advantage": success_low > 0.0,
            }
        )
    return rows


def episode_rows(episode: EpisodeResult) -> list[dict[str, object]]:
    """Convert a demo episode into long-form rows."""

    rows: list[dict[str, object]] = []
    for step_index in range(len(episode.beliefs)):
        diag = extract_step_diagnostics(episode, step_index)
        row = {
            "Step": int(step_index),
            "Observation": observation_label(diag.observation),
            "Hidden State": state_label(diag.hidden_state),
            "Most Likely State": state_label(int(np.argmax(diag.belief))),
            "Posterior Confidence": float(np.max(diag.belief)),
            "Posterior Entropy": float(diag.posterior_entropy),
            "Free Energy": float(diag.free_energy),
        }
        if diag.selected_action is not None:
            row.update(
                {
                    "Action": action_label(diag.selected_action),
                    "Action Probability": float(diag.selected_action_probability or 0.0),
                    "Pragmatic": float(diag.pragmatic_values[diag.selected_action]),
                    "Epistemic": float(diag.epistemic_values[diag.selected_action]),
                    "Total EFE": float(diag.efe_values[diag.selected_action]),
                }
            )
        rows.append(row)
    return rows


def rows_to_csv_text(rows: Sequence[dict[str, object]], fieldnames: Sequence[str] | None = None) -> str:
    """Serialize rows to CSV text."""

    if not rows:
        return ""
    headers = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def rows_to_markdown(rows: Sequence[dict[str, object]], fieldnames: Sequence[str] | None = None) -> str:
    """Render a small markdown table."""

    if not rows:
        return "_No rows to display._"
    headers = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines: list[str] = []
    for row in rows:
        values = [_cell_to_text(row.get(header, "")) for header in headers]
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header_line, separator_line, *body_lines])


def _cell_to_text(value: object) -> str:
    if isinstance(value, float):
        return format_float(value)
    if isinstance(value, (np.floating,)):
        return format_float(float(value))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return ""
    return str(value)

