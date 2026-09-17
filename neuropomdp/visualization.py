"""Matplotlib visualization helpers for NeuroPOMDP results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import ActionID, BenchmarkResult, EpisodeResult, MultiSeedSweepResult, SweepResult


def _prepare_backend() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)


def _save(fig, path: Path | str | None) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")


def _action_label(action: int) -> str:
    mapping = {
        int(ActionID.INSPECT): "inspect",
        int(ActionID.EXPLOIT_LEFT): "left",
        int(ActionID.EXPLOIT_RIGHT): "right",
    }
    return mapping.get(int(action), str(action))


def plot_state_and_belief(episode: EpisodeResult, path: Path | str | None = None):
    """Figure 1: true state trajectory and posterior confidence."""

    _prepare_backend()
    import matplotlib.pyplot as plt

    beliefs = np.asarray(episode.beliefs, dtype=np.float64)
    hidden_states = np.asarray(episode.hidden_states, dtype=np.int32)
    time = np.arange(len(hidden_states))
    true_state_prob = beliefs[np.arange(len(hidden_states)), hidden_states]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].step(time, hidden_states, where="post", color="black", linewidth=2)
    axes[0].set_ylabel("True state")
    axes[0].set_title("Hidden state and inferred posterior")
    axes[1].plot(time, true_state_prob, marker="o", color="#1f77b4")
    axes[1].set_ylabel("Posterior P(true state)")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylim(0.0, 1.0)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_posterior_entropy(episode: EpisodeResult, path: Path | str | None = None):
    """Figure 2: posterior entropy over time."""

    _prepare_backend()
    import matplotlib.pyplot as plt

    time = np.arange(len(episode.posterior_entropies))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(time, episode.posterior_entropies, marker="o", color="#d62728")
    ax.set_title("Posterior entropy over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Entropy (nats)")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_actions_over_time(episode: EpisodeResult, path: Path | str | None = None):
    """Figure 3: selected actions over time."""

    _prepare_backend()
    import matplotlib.pyplot as plt

    time = np.arange(len(episode.actions))
    labels = [_action_label(action) for action in episode.actions]
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.step(time, episode.actions, where="post", color="#2ca02c", linewidth=2)
    ax.scatter(time, episode.actions, color="#2ca02c")
    ax.set_yticks([int(ActionID.INSPECT), int(ActionID.EXPLOIT_LEFT), int(ActionID.EXPLOIT_RIGHT)])
    ax.set_yticklabels(["inspect", "left", "right"])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Chosen action")
    ax.set_title(f"Selected actions: {', '.join(labels) if labels else 'none'}")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_pragmatic_vs_epistemic(episode: EpisodeResult, path: Path | str | None = None):
    """Figure 4: pragmatic and epistemic value for candidate actions."""

    _prepare_backend()
    import matplotlib.pyplot as plt

    pragmatic = np.asarray(episode.pragmatic_values[0], dtype=np.float64)
    epistemic = np.asarray(episode.epistemic_values[0], dtype=np.float64)
    actions = np.arange(len(pragmatic))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(actions - width / 2, pragmatic, width=width, label="pragmatic cost", color="#ff7f0e")
    ax.bar(actions + width / 2, epistemic, width=width, label="epistemic value", color="#1f77b4")
    ax.set_xticks(actions)
    ax.set_xticklabels(["inspect", "left", "right"])
    ax.set_ylabel("Value")
    ax.set_title("Candidate action components")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_efe_values(episode: EpisodeResult, path: Path | str | None = None):
    """Figure 5: total EFE for candidate actions."""

    _prepare_backend()
    import matplotlib.pyplot as plt

    efe = np.asarray(episode.efe_values[0], dtype=np.float64)
    actions = np.arange(len(efe))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(actions, efe, color="#9467bd")
    ax.set_xticks(actions)
    ax.set_xticklabels(["inspect", "left", "right"])
    ax.set_ylabel("Expected Free Energy")
    ax.set_title("Expected Free Energy by action")
    fig.tight_layout()
    _save(fig, path)
    return fig


def _cumulative_reward_summary(episodes: list[EpisodeResult]) -> tuple[np.ndarray, np.ndarray]:
    trajectories = []
    max_length = max((len(episode.rewards) - 1 for episode in episodes), default=0)
    for episode in episodes:
        rewards = np.asarray(episode.rewards[1:], dtype=np.float64)
        cumulative = np.cumsum(rewards)
        padded = np.full((max_length,), np.nan, dtype=np.float64)
        padded[: len(cumulative)] = cumulative
        trajectories.append(padded)
    if not trajectories:
        return np.asarray([]), np.asarray([])

    trajectory_array = np.asarray(trajectories, dtype=np.float64)
    mean_curve = np.nanmean(trajectory_array, axis=0)
    std_curve = np.nanstd(trajectory_array, axis=0, ddof=0)
    valid_counts = np.sum(~np.isnan(trajectory_array), axis=0)
    margin = np.where(valid_counts > 1, 1.96 * std_curve / np.sqrt(valid_counts), 0.0)
    return mean_curve, margin


def plot_benchmark_cumulative_reward(result: BenchmarkResult, path: Path | str | None = None):
    """Figure 6: cumulative reward for the benchmark agents."""

    _prepare_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for agent_name, episodes in result.episodes_by_agent.items():
        mean_curve, ci95 = _cumulative_reward_summary(episodes)
        if mean_curve.size == 0:
            continue
        x = np.arange(1, len(mean_curve) + 1)
        ax.errorbar(x, mean_curve, yerr=ci95, fmt="-o", capsize=3, label=agent_name, alpha=0.8)
    ax.set_xlabel("Action step")
    ax.set_ylabel("Mean cumulative reward")
    ax.set_title("Benchmark cumulative reward")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_ablation(result: BenchmarkResult, path: Path | str | None = None):
    """Figure 7: Active Inference versus no-epistemic ablation."""

    _prepare_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for agent_name, episodes in result.episodes_by_agent.items():
        mean_curve, ci95 = _cumulative_reward_summary(episodes)
        if mean_curve.size == 0:
            continue
        x = np.arange(1, len(mean_curve) + 1)
        ax.errorbar(x, mean_curve, yerr=ci95, fmt="-o", capsize=3, label=agent_name, alpha=0.8)
    ax.set_xlabel("Action step")
    ax.set_ylabel("Mean cumulative reward")
    ax.set_title("Ablation study")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_parameter_sweep(
    result: SweepResult | MultiSeedSweepResult,
    path: Path | str | None = None,
):
    """Figure 8: performance as observation uncertainty changes."""

    _prepare_backend()
    import matplotlib.pyplot as plt

    noise = np.asarray(result.noise_values, dtype=np.float64)
    rewards = np.asarray(result.mean_reward, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    seed_rewards = None
    if isinstance(result, MultiSeedSweepResult) and result.per_seed_results:
        seed_rewards = np.asarray(
            [seed_result.mean_reward for seed_result in result.per_seed_results],
            dtype=np.float64,
        )
    for agent_index, agent_name in enumerate(result.agent_names):
        if seed_rewards is not None and seed_rewards.shape[0] > 1:
            values = seed_rewards[:, :, agent_index]
            std = np.std(values, axis=0, ddof=0)
            margin = 1.96 * std / np.sqrt(values.shape[0])
            ax.errorbar(
                noise,
                rewards[:, agent_index],
                yerr=margin,
                marker="o",
                capsize=3,
                label=agent_name,
            )
        else:
            ax.plot(noise, rewards[:, agent_index], marker="o", label=agent_name)
    ax.set_xlabel("Observation noise")
    ax.set_ylabel("Mean reward")
    ax.set_title("Parameter sweep")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path)
    return fig

