"""Streamlit rendering helpers for the NeuroPOMDP dashboard."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from ..types import EpisodeResult
from .formatting import ACTION_LABELS, STATE_LABELS, StepDiagnostics, action_label, format_float, state_label
from .i18n import t


def figure_to_png_bytes(fig: plt.Figure) -> bytes:
    """Serialize a Matplotlib figure to PNG bytes."""

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def build_hidden_state_figure(episode: EpisodeResult) -> plt.Figure:
    """Create the hidden-state beliefs chart."""

    beliefs = np.asarray(episode.beliefs, dtype=np.float64)
    hidden_states = np.asarray(episode.hidden_states, dtype=np.int32)
    time = np.arange(len(beliefs))

    fig, (ax_state, ax_belief) = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
    ax_state.step(time, hidden_states, where="post", color="#111111", linewidth=2.0, label=t("chart_true_state"))
    ax_state.set_yticks(np.arange(len(STATE_LABELS)))
    ax_state.set_yticklabels([t(f"state_{index}") for index in range(len(STATE_LABELS))], fontsize=8)
    ax_state.set_ylabel(t("chart_true_state_ylabel"))
    ax_state.set_title(t("chart_hidden_state_beliefs"))
    ax_state.grid(True, alpha=0.15)

    for state_index in range(beliefs.shape[1]):
        ax_belief.plot(
            time,
            beliefs[:, state_index],
            linewidth=2.0,
            label=t(f"state_{state_index}"),
        )
    ax_belief.set_ylim(0.0, 1.0)
    ax_belief.set_xlabel(t("chart_time_step"))
    ax_belief.set_ylabel(t("chart_posterior_probability"))
    ax_belief.grid(True, alpha=0.15)
    ax_belief.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def build_entropy_figure(episode: EpisodeResult) -> plt.Figure:
    """Create the posterior entropy chart."""

    time = np.arange(len(episode.posterior_entropies))
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(time, episode.posterior_entropies, color="#c44e52", linewidth=2.0, marker="o")
    ax.set_title(t("chart_posterior_entropy"))
    ax.set_xlabel(t("chart_time_step"))
    ax.set_ylabel(t("chart_entropy_nats"))
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def build_actions_figure(episode: EpisodeResult) -> plt.Figure:
    """Create the actions-over-time chart."""

    time = np.arange(len(episode.actions))
    action_names = [t(f"action_{action}") for action in episode.actions]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.step(time, episode.actions, where="post", color="#4c72b0", linewidth=2.0)
    ax.scatter(time, episode.actions, color="#4c72b0")
    ax.set_yticks(np.arange(len(ACTION_LABELS)))
    ax.set_yticklabels([t(f"action_{index}") for index in range(len(ACTION_LABELS))])
    ax.set_title(t("chart_actions_over_time"))
    ax.set_xlabel(t("chart_time_step"))
    ax.set_ylabel(t("chart_selected_action"))
    ax.grid(True, alpha=0.15)
    ax.text(
        0.01,
        0.02,
        ", ".join(action_names),
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    return fig


def build_value_comparison_figure(step: StepDiagnostics) -> plt.Figure:
    """Create the pragmatic vs epistemic value chart."""

    actions = np.arange(len(step.efe_values))
    width = 0.28
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(actions - width, step.pragmatic_values, width=width, label=t("pragmatic"), color="#dd8452")
    ax.bar(actions, step.epistemic_values, width=width, label=t("epistemic"), color="#55a868")
    ax.bar(actions + width, step.efe_values, width=width, label=t("total_efe"), color="#4c72b0")
    ax.set_xticks(actions)
    ax.set_xticklabels([t(f"action_{index}") for index in range(len(step.efe_values))])
    ax.set_ylabel(t("chart_value"))
    ax.set_title(t("chart_pragmatic_epistemic"))
    ax.grid(True, axis="y", alpha=0.15)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def build_efe_figure(step: StepDiagnostics) -> plt.Figure:
    """Create the expected free energy by action chart."""

    actions = np.arange(len(step.efe_values))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(actions - width, step.pragmatic_values, width=width, label=t("pragmatic"), color="#dd8452")
    ax.bar(actions, step.epistemic_values, width=width, label=t("epistemic"), color="#55a868")
    ax.bar(actions + width, step.efe_values, width=width, label=t("total_efe"), color="#8172b2")
    ax.set_xticks(actions)
    ax.set_xticklabels([t(f"action_{index}") for index in range(len(step.efe_values))])
    ax.set_ylabel(t("chart_value"))
    ax.set_title(t("chart_expected_free_energy"))
    ax.grid(True, axis="y", alpha=0.15)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def build_action_probability_figure(step: StepDiagnostics) -> plt.Figure:
    """Create a bar chart of action probabilities."""

    actions = np.arange(len(step.action_probabilities))
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.bar(actions, step.action_probabilities, color="#4c72b0")
    ax.set_xticks(actions)
    ax.set_xticklabels([t(f"action_{index}") for index in range(len(step.action_probabilities))])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(t("chart_probability"))
    ax.set_title(t("chart_action_probabilities"))
    ax.grid(True, axis="y", alpha=0.15)
    fig.tight_layout()
    return fig


def build_model_matrix_figure(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
) -> plt.Figure:
    """Render a labeled heatmap for the generative model."""

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = float(matrix[row_index, col_index])
            if value >= 0.05:
                ax.text(
                    col_index,
                    row_index,
                    format_float(value, 2),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def build_model_vector_figure(values: np.ndarray, labels: list[str], title: str, ylabel: str) -> plt.Figure:
    """Render a one-dimensional model vector as a bar chart."""

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.bar(np.arange(len(values)), values, color="#4c72b0")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.15)
    fig.tight_layout()
    return fig

