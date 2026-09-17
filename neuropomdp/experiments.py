"""Ablation and parameter sweep experiments."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from .agent import ActiveInferenceAgent, PragmaticOnlyAgent
from .config import BenchmarkConfig
from .model import build_generative_model
from .simulation import AgentLike, run_benchmark
from .types import BenchmarkResult, MultiSeedSweepResult, SweepResult


def _active_only_factories(config: BenchmarkConfig) -> list[tuple[str, Callable[[], AgentLike]]]:
    model = build_generative_model(config.env.model)
    active_selection = replace(config.selection, use_epistemic=True)
    pragmatic_selection = replace(config.selection, use_epistemic=False, planning_horizon=1)
    return [
        (
            "active_inference",
            lambda: ActiveInferenceAgent(
                model=model,
                selection_config=active_selection,
                inference_config=config.inference,
            ),
        ),
        (
            "no_epistemic",
            lambda: PragmaticOnlyAgent(
                model=model,
                selection_config=pragmatic_selection,
                inference_config=config.inference,
            ),
        ),
    ]


def run_ablation_study(config: BenchmarkConfig) -> BenchmarkResult:
    """Compare full Active Inference against the no-epistemic ablation."""

    factories = _active_only_factories(config)
    return run_benchmark(factories, config.simulation, config.env)


def run_parameter_sweep(config: BenchmarkConfig) -> SweepResult:
    """Sweep observation uncertainty and compare the two key agents."""

    noise_values = list(config.sweep_noise_values)
    agent_names = ["active_inference", "no_epistemic"]
    mean_reward: list[list[float]] = []
    success_rate: list[list[float]] = []
    mean_entropy: list[list[float]] = []
    info_action_frequency: list[list[float]] = []

    for noise in noise_values:
        swept_model = replace(config.env.model, observation_noise=float(noise))
        swept_env = replace(config.env, model=swept_model)
        swept_config = replace(config, env=swept_env)
        benchmark = run_benchmark(_active_only_factories(swept_config), swept_config.simulation, swept_config.env)
        summary_map = {summary.agent_name: summary for summary in benchmark.summaries}
        mean_reward.append([summary_map[name].mean_reward for name in agent_names])
        success_rate.append([summary_map[name].success_rate for name in agent_names])
        mean_entropy.append([summary_map[name].mean_posterior_entropy for name in agent_names])
        info_action_frequency.append([summary_map[name].info_action_frequency for name in agent_names])

    return SweepResult(
        seed=config.simulation.seed,
        config=config,
        noise_values=noise_values,
        agent_names=agent_names,
        mean_reward=mean_reward,
        success_rate=success_rate,
        mean_entropy=mean_entropy,
        info_action_frequency=info_action_frequency,
    )


def run_ablation_study_multi_seed(
    config: BenchmarkConfig,
    seeds: tuple[int, ...] | list[int],
) -> list[BenchmarkResult]:
    """Run the active-vs-no-epistemic ablation independently for each seed."""

    normalized_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not normalized_seeds:
        raise ValueError("At least one seed is required.")
    return [
        run_ablation_study(replace(config, simulation=replace(config.simulation, seed=seed)))
        for seed in normalized_seeds
    ]


def run_parameter_sweep_multi_seed(
    config: BenchmarkConfig,
    seeds: tuple[int, ...] | list[int],
) -> list[SweepResult]:
    """Run the parameter sweep independently for each seed."""

    normalized_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not normalized_seeds:
        raise ValueError("At least one seed is required.")
    return [
        run_parameter_sweep(replace(config, simulation=replace(config.simulation, seed=seed)))
        for seed in normalized_seeds
    ]


def aggregate_sweep_results(results: list[SweepResult]) -> MultiSeedSweepResult:
    """Aggregate multiple sweep runs by noise level and agent across seeds."""

    if not results:
        raise ValueError("At least one sweep result is required.")

    noise_values = results[0].noise_values
    agent_names = results[0].agent_names
    if any(result.noise_values != noise_values for result in results[1:]):
        raise ValueError("Sweep results must share the same noise values for aggregation.")
    if any(result.agent_names != agent_names for result in results[1:]):
        raise ValueError("Sweep results must share the same agent names for aggregation.")

    def _aggregate_matrix(matrix_list: list[list[list[float]]]) -> list[list[float]]:
        noise_count = len(noise_values)
        agent_count = len(agent_names)
        aggregated = [[0.0 for _ in range(agent_count)] for _ in range(noise_count)]
        for noise_idx in range(noise_count):
            for agent_idx in range(agent_count):
                values = [run[noise_idx][agent_idx] for run in matrix_list]
                aggregated[noise_idx][agent_idx] = float(sum(values) / len(values))
        return aggregated

    aggregated = MultiSeedSweepResult(
        seed_count=len(results),
        seeds=[int(result.seed) for result in results],
        config=results[0].config,
        noise_values=noise_values,
        agent_names=agent_names,
        mean_reward=_aggregate_matrix([result.mean_reward for result in results]),
        success_rate=_aggregate_matrix([result.success_rate for result in results]),
        mean_entropy=_aggregate_matrix([result.mean_entropy for result in results]),
        info_action_frequency=_aggregate_matrix([result.info_action_frequency for result in results]),
        per_seed_results=list(results),
    )
    return aggregated


def summarize_ablation_effect(results: list[BenchmarkResult]) -> dict[str, float | int | str]:
    """Summarize the effect of epistemic value across repeated ablation runs."""

    if not results:
        raise ValueError("At least one ablation result is required.")

    agent_names = results[0].agent_names
    if len(agent_names) < 2:
        raise ValueError("Ablation results must include both active and baseline agents.")

    active_name = "active_inference"
    baseline_name = "no_epistemic"
    if active_name not in agent_names or baseline_name not in agent_names:
        active_name = agent_names[0]
        baseline_name = agent_names[1]

    active_rewards = []
    baseline_rewards = []
    active_successes = []
    baseline_successes = []

    for result in results:
        summary_map = {summary.agent_name: summary for summary in result.summaries}
        active_rewards.append(float(summary_map[active_name].mean_reward))
        baseline_rewards.append(float(summary_map[baseline_name].mean_reward))
        active_successes.append(float(summary_map[active_name].success_rate))
        baseline_successes.append(float(summary_map[baseline_name].success_rate))

    reward_delta = float(sum(active_rewards) / len(active_rewards) - sum(baseline_rewards) / len(baseline_rewards))
    success_delta = float(sum(active_successes) / len(active_successes) - sum(baseline_successes) / len(baseline_successes))
    reward_deltas = [a - b for a, b in zip(active_rewards, baseline_rewards, strict=False)]
    success_deltas = [a - b for a, b in zip(active_successes, baseline_successes, strict=False)]
    reward_mean = float(sum(reward_deltas) / len(reward_deltas))
    reward_std = float((sum((value - reward_mean) ** 2 for value in reward_deltas) / len(reward_deltas)) ** 0.5)
    success_mean = float(sum(success_deltas) / len(success_deltas))
    success_std = float((sum((value - success_mean) ** 2 for value in success_deltas) / len(success_deltas)) ** 0.5)
    reward_margin = 1.96 * reward_std / (len(reward_deltas) ** 0.5) if len(reward_deltas) > 1 else 0.0
    success_margin = 1.96 * success_std / (len(success_deltas) ** 0.5) if len(success_deltas) > 1 else 0.0

    return {
        "seed_count": len(results),
        "active_agent": active_name,
        "baseline_agent": baseline_name,
        "reward_delta": reward_delta,
        "reward_delta_ci95_low": reward_mean - reward_margin,
        "reward_delta_ci95_high": reward_mean + reward_margin,
        "success_delta": success_delta,
        "success_delta_ci95_low": success_mean - success_margin,
        "success_delta_ci95_high": success_mean + success_margin,
    }


def summarize_sweep_effect(results: list[SweepResult]) -> dict[str, float | int | str | list[float]]:
    """Summarize the per-noise benefit of active inference over the no-epistemic baseline."""

    if not results:
        raise ValueError("At least one sweep result is required.")

    noise_values = results[0].noise_values
    agent_names = results[0].agent_names
    if any(result.noise_values != noise_values for result in results[1:]):
        raise ValueError("Sweep results must share the same noise values for aggregation.")

    active_idx = agent_names.index("active_inference") if "active_inference" in agent_names else 0
    baseline_idx = agent_names.index("no_epistemic") if "no_epistemic" in agent_names else 1 if len(agent_names) > 1 else 0

    reward_deltas_by_noise: list[float] = []
    success_deltas_by_noise: list[float] = []
    reward_ci95_low_by_noise: list[float] = []
    reward_ci95_high_by_noise: list[float] = []
    success_ci95_low_by_noise: list[float] = []
    success_ci95_high_by_noise: list[float] = []
    for noise_idx, _ in enumerate(noise_values):
        per_seed_reward_deltas: list[float] = []
        per_seed_success_deltas: list[float] = []
        for result in results:
            active_reward = result.mean_reward[noise_idx][active_idx]
            baseline_reward = result.mean_reward[noise_idx][baseline_idx]
            active_success = result.success_rate[noise_idx][active_idx]
            baseline_success = result.success_rate[noise_idx][baseline_idx]
            per_seed_reward_deltas.append(active_reward - baseline_reward)
            per_seed_success_deltas.append(active_success - baseline_success)
        reward_mean = float(sum(per_seed_reward_deltas) / len(per_seed_reward_deltas))
        success_mean = float(sum(per_seed_success_deltas) / len(per_seed_success_deltas))
        reward_std = float(np.std(per_seed_reward_deltas, ddof=0))
        success_std = float(np.std(per_seed_success_deltas, ddof=0))
        reward_margin = 1.96 * reward_std / np.sqrt(len(per_seed_reward_deltas)) if len(per_seed_reward_deltas) > 1 else 0.0
        success_margin = 1.96 * success_std / np.sqrt(len(per_seed_success_deltas)) if len(per_seed_success_deltas) > 1 else 0.0
        reward_deltas_by_noise.append(reward_mean)
        success_deltas_by_noise.append(success_mean)
        reward_ci95_low_by_noise.append(reward_mean - reward_margin)
        reward_ci95_high_by_noise.append(reward_mean + reward_margin)
        success_ci95_low_by_noise.append(success_mean - success_margin)
        success_ci95_high_by_noise.append(success_mean + success_margin)

    return {
        "seed_count": len(results),
        "active_agent": agent_names[active_idx],
        "baseline_agent": agent_names[baseline_idx],
        "noise_values": noise_values,
        "reward_deltas": reward_deltas_by_noise,
        "success_deltas": success_deltas_by_noise,
        "reward_delta_ci95_low": reward_ci95_low_by_noise,
        "reward_delta_ci95_high": reward_ci95_high_by_noise,
        "success_delta_ci95_low": success_ci95_low_by_noise,
        "success_delta_ci95_high": success_ci95_high_by_noise,
        "reliably_reward_better_noise_values": [
            noise
            for noise, lower in zip(noise_values, reward_ci95_low_by_noise, strict=False)
            if lower > 0.0
        ],
        "reliably_success_better_noise_values": [
            noise
            for noise, lower in zip(noise_values, success_ci95_low_by_noise, strict=False)
            if lower > 0.0
        ],
    }
