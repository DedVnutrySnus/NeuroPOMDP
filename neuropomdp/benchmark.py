"""Benchmark orchestration for the NeuroPOMDP agents."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

import numpy as np

from .agent import ActiveInferenceAgent, BayesianUtilityAgent, PragmaticOnlyAgent, RandomAgent
from .config import BenchmarkConfig
from .model import build_generative_model
from .simulation import AgentLike, run_benchmark
from .types import AggregateBenchmarkResult, AgentSummary, BenchmarkResult, MultiSeedBenchmarkResult


def default_agent_factories(config: BenchmarkConfig) -> list[tuple[str, Callable[[], AgentLike]]]:
    """Construct the default set of benchmark agents."""

    model = build_generative_model(config.env.model)
    active_selection = replace(config.selection, use_epistemic=True)
    pragmatic_selection = replace(config.selection, use_epistemic=False)

    return [
        ("random", lambda: RandomAgent(model=model)),
        (
            "pragmatic_only",
            lambda: PragmaticOnlyAgent(
                model=model,
                selection_config=pragmatic_selection,
                inference_config=config.inference,
            ),
        ),
        (
            "active_inference",
            lambda: ActiveInferenceAgent(
                model=model,
                selection_config=active_selection,
                inference_config=config.inference,
            ),
        ),
        (
            "bayesian_utility",
            lambda: BayesianUtilityAgent(
                model=model,
                selection_config=pragmatic_selection,
                inference_config=config.inference,
            ),
        ),
    ]


def benchmark_agents(config: BenchmarkConfig) -> BenchmarkResult:
    """Run the default benchmark suite."""

    factories = default_agent_factories(config)
    return run_benchmark(factories, config.simulation, config.env)


def benchmark_agents_multi_seed(
    config: BenchmarkConfig,
    seeds: tuple[int, ...] | list[int],
) -> list[BenchmarkResult]:
    """Run the default benchmark suite independently for each seed."""

    normalized_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not normalized_seeds:
        raise ValueError("At least one seed is required.")
    return [
        benchmark_agents(replace(config, simulation=replace(config.simulation, seed=seed)))
        for seed in normalized_seeds
    ]


def summarize_aggregate_agent(agent_name: str, results: Sequence[BenchmarkResult]) -> AgentSummary:
    """Aggregate one agent across multiple benchmark seeds."""

    matched = [next(summary for summary in result.summaries if summary.agent_name == agent_name) for result in results]
    if not matched:
        raise ValueError(f"No results found for agent {agent_name!r}.")

    reward_values = np.asarray([summary.mean_reward for summary in matched], dtype=np.float64)
    success_values = np.asarray([summary.success_rate for summary in matched], dtype=np.float64)
    entropy_values = np.asarray([summary.mean_posterior_entropy for summary in matched], dtype=np.float64)
    info_values = np.asarray([summary.mean_information_gain for summary in matched], dtype=np.float64)
    info_frequency = np.asarray([summary.info_action_frequency for summary in matched], dtype=np.float64)
    episode_count = int(sum(summary.episode_count for summary in matched))
    mean_reward = float(reward_values.mean()) if reward_values.size else 0.0
    reward_std = float(reward_values.std(ddof=0)) if reward_values.size else 0.0
    reward_margin = 1.96 * reward_std / np.sqrt(reward_values.size) if reward_values.size > 1 else 0.0
    success_rate = float(success_values.mean()) if success_values.size else 0.0
    success_margin = 1.96 * np.sqrt(success_rate * (1.0 - success_rate) / max(1, success_values.size))
    action_counts = {}
    for summary in matched:
        for key, value in summary.action_counts.items():
            action_counts[key] = action_counts.get(key, 0) + int(value)

    return AgentSummary(
        agent_name=agent_name,
        episode_count=episode_count,
        mean_reward=mean_reward,
        reward_std=reward_std,
        reward_ci95_low=mean_reward - reward_margin,
        reward_ci95_high=mean_reward + reward_margin,
        success_rate=success_rate,
        success_ci95_low=max(0.0, success_rate - success_margin),
        success_ci95_high=min(1.0, success_rate + success_margin),
        mean_posterior_entropy=float(entropy_values.mean()) if entropy_values.size else 0.0,
        mean_information_gain=float(info_values.mean()) if info_values.size else 0.0,
        info_action_frequency=float(info_frequency.mean()) if info_frequency.size else 0.0,
        action_counts=action_counts,
    )


def aggregate_benchmark_results(results: Sequence[BenchmarkResult]) -> MultiSeedBenchmarkResult:
    """Aggregate multiple independent benchmark results into one summary."""

    normalized = list(results)
    if not normalized:
        raise ValueError("At least one benchmark result is required.")
    agent_names = normalized[0].agent_names
    if any(result.agent_names != agent_names for result in normalized[1:]):
        agent_names = list(dict.fromkeys(name for result in normalized for name in result.agent_names))
    summaries = [summarize_aggregate_agent(agent_name, normalized) for agent_name in agent_names]
    aggregate = MultiSeedBenchmarkResult(
        seed_count=len(normalized),
        seeds=[int(result.seed) for result in normalized],
        config=normalized[0].config,
        agent_names=agent_names,
        summaries=summaries,
        per_seed_results=list(normalized),
    )
    return aggregate


__all__ = [
    "AggregateBenchmarkResult",
    "MultiSeedBenchmarkResult",
    "aggregate_benchmark_results",
    "benchmark_agents",
    "benchmark_agents_multi_seed",
    "default_agent_factories",
]
