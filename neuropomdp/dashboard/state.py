"""Experiment orchestration helpers for the dashboard."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
from typing import Callable

import jax
import numpy as np

from ..agent import ActiveInferenceAgent, PragmaticOnlyAgent, RandomAgent
from ..config import BenchmarkConfig
from ..model import build_generative_model
from ..simulation import AgentLike, run_episode, summarize_agent
from ..experiments import aggregate_sweep_results
from ..types import BenchmarkResult, EpisodeResult, MultiSeedSweepResult, SweepResult, to_serializable


ProgressCallback = Callable[[int, int, str], None]


def get_default_output_dir() -> str:
    """Return a writable default output directory for the desktop dashboard."""

    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return str(Path(base) / "NeuroPOMDP" / "outputs")


def build_dashboard_config(
    *,
    seed: int,
    episodes: int,
    episode_length: int,
    output_dir: str,
    show_plots: bool,
    observation_noise: float,
    information_quality: float,
    transition_stochasticity: float,
    initial_uncertainty: float,
    precision: float,
    deterministic: bool,
    inference_eps: float,
    enable_x64: bool = False,
) -> BenchmarkConfig:
    """Build a benchmark config from sidebar controls."""

    base = BenchmarkConfig()
    resolved_output_dir = output_dir.strip() if output_dir.strip() else get_default_output_dir()
    model = replace(
        base.env.model,
        observation_noise=float(observation_noise),
        information_quality=float(information_quality),
        transition_stochasticity=float(transition_stochasticity),
        prior_left_prob=float(initial_uncertainty),
        enable_x64=bool(enable_x64),
        dtype="float64" if enable_x64 else base.env.model.dtype,
    )
    env = replace(base.env, model=model, episode_length=int(episode_length))
    simulation = replace(
        base.simulation,
        seed=int(seed),
        episodes=int(episodes),
        episode_length=int(episode_length),
        enable_plots=bool(show_plots),
        output_dir=resolved_output_dir,
    )
    selection = replace(
        base.selection,
        precision=float(precision),
        deterministic=bool(deterministic),
    )
    inference = replace(base.inference, eps=float(inference_eps))
    return BenchmarkConfig(
        env=env,
        inference=inference,
        selection=selection,
        simulation=simulation,
        sweep_noise_values=base.sweep_noise_values,
    )


def run_demo(config: BenchmarkConfig) -> EpisodeResult:
    """Run one active inference episode for the dashboard demo."""

    model = build_generative_model(config.env.model)
    agent = ActiveInferenceAgent(
        model=model,
        inference_config=config.inference,
        selection_config=config.selection,
    )
    return run_episode(agent, config.env, jax.random.PRNGKey(config.simulation.seed))


def _episode_key(seed: int, episode_index: int) -> jax.Array:
    return jax.random.PRNGKey(seed + episode_index * 9973)


def _build_factories(config: BenchmarkConfig) -> list[tuple[str, Callable[[], AgentLike]]]:
    """Create the dashboard's benchmark agents."""

    model = build_generative_model(config.env.model)
    active_selection = replace(config.selection, use_epistemic=True)
    pragmatic_selection = replace(config.selection, use_epistemic=False)

    return [
        ("random", lambda model=model: RandomAgent(model=model)),
        (
            "pragmatic_only",
            lambda model=model, selection=pragmatic_selection: PragmaticOnlyAgent(
                model=model,
                selection_config=selection,
                inference_config=config.inference,
            ),
        ),
        (
            "active_inference",
            lambda model=model, selection=active_selection: ActiveInferenceAgent(
                model=model,
                selection_config=selection,
                inference_config=config.inference,
            ),
        ),
    ]


def _run_factories(
    config: BenchmarkConfig,
    factories: list[tuple[str, Callable[[], AgentLike]]],
    progress_callback: ProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> BenchmarkResult:
    """Run a sequence of agent factories and collect benchmark summaries."""

    total_episodes = len(factories) * int(config.simulation.episodes)
    total = int(progress_total) if progress_total is not None else total_episodes
    completed = int(progress_offset)
    episodes_by_agent: dict[str, list[EpisodeResult]] = {}
    summaries = []

    for agent_name, factory in factories:
        episodes: list[EpisodeResult] = []
        for episode_index in range(int(config.simulation.episodes)):
            episode = run_episode(factory(), config.env, _episode_key(config.simulation.seed, episode_index))
            episodes.append(episode)
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    completed,
                    total,
                    f"Running {agent_name} episode {episode_index + 1} of {config.simulation.episodes}",
                )
        episodes_by_agent[agent_name] = episodes
        summaries.append(summarize_agent(agent_name, episodes))

    return BenchmarkResult(
        seed=config.simulation.seed,
        config={"simulation": to_serializable(config.simulation), "environment": to_serializable(config.env)},
        agent_names=[agent_name for agent_name, _ in factories],
        summaries=summaries,
        episodes_by_agent=episodes_by_agent,
    )


def run_benchmark_dashboard(
    config: BenchmarkConfig,
    progress_callback: ProgressCallback | None = None,
) -> BenchmarkResult:
    """Run the benchmark view used by the dashboard."""

    return _run_factories(config, _build_factories(config), progress_callback=progress_callback)


def run_ablation_dashboard(config: BenchmarkConfig) -> BenchmarkResult:
    """Run the two-agent ablation view used by the dashboard."""

    model = build_generative_model(config.env.model)
    active_selection = replace(config.selection, use_epistemic=True)
    pragmatic_selection = replace(config.selection, use_epistemic=False)
    factories = [
        (
            "active_inference",
            lambda model=model, selection=active_selection: ActiveInferenceAgent(
                model=model,
                selection_config=selection,
                inference_config=config.inference,
            ),
        ),
        (
            "no_epistemic",
            lambda model=model, selection=pragmatic_selection: PragmaticOnlyAgent(
                model=model,
                selection_config=selection,
                inference_config=config.inference,
            ),
        ),
    ]
    return _run_factories(config, factories)


def run_sweep_dashboard(
    config: BenchmarkConfig,
    noise_min: float,
    noise_max: float,
    sweep_points: int,
    episodes_per_point: int,
    progress_callback: ProgressCallback | None = None,
) -> SweepResult:
    """Run a sweep over observation noise for the dashboard."""

    if sweep_points < 1:
        raise ValueError("sweep_points must be at least 1.")
    if episodes_per_point < 1:
        raise ValueError("episodes_per_point must be at least 1.")

    noise_values = np.linspace(float(noise_min), float(noise_max), int(sweep_points)).tolist()
    agent_names = ["random", "pragmatic_only", "active_inference"]
    mean_reward: list[list[float]] = []
    success_rate: list[list[float]] = []
    mean_entropy: list[list[float]] = []
    info_action_frequency: list[list[float]] = []

    total_tasks = len(noise_values)
    for noise_index, noise in enumerate(noise_values):
        swept_model = replace(config.env.model, observation_noise=float(noise))
        swept_config = replace(
            config,
            env=replace(config.env, model=swept_model),
            simulation=replace(config.simulation, episodes=int(episodes_per_point)),
        )
        benchmark = _run_factories(swept_config, _build_factories(swept_config))
        summary_map = {summary.agent_name: summary for summary in benchmark.summaries}
        mean_reward.append([summary_map[name].mean_reward for name in agent_names])
        success_rate.append([summary_map[name].success_rate for name in agent_names])
        mean_entropy.append([summary_map[name].mean_posterior_entropy for name in agent_names])
        info_action_frequency.append([summary_map[name].info_action_frequency for name in agent_names])
        if progress_callback is not None:
            progress_callback(
                noise_index + 1,
                total_tasks,
                f"Completed sweep point {noise_index + 1} of {total_tasks} at noise={noise:.3f}",
            )

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


def run_sweep_dashboard_multi_seed(
    config: BenchmarkConfig,
    noise_min: float,
    noise_max: float,
    sweep_points: int,
    episodes_per_point: int,
    seeds: tuple[int, ...] | list[int],
    progress_callback: ProgressCallback | None = None,
) -> MultiSeedSweepResult:
    """Run and aggregate the dashboard sweep across independent seeds."""

    normalized_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not normalized_seeds:
        raise ValueError("At least one sweep seed is required.")

    total_points = len(normalized_seeds) * int(sweep_points)
    results: list[SweepResult] = []
    for seed_index, seed in enumerate(normalized_seeds):
        seeded_config = replace(config, simulation=replace(config.simulation, seed=seed))

        def update_seed_progress(done: int, total: int, message: str, offset=seed_index * int(sweep_points)) -> None:
            if progress_callback is not None:
                progress_callback(offset + int(done), total_points, f"Seed {seed}: {message}")

        results.append(
            run_sweep_dashboard(
                seeded_config,
                noise_min=noise_min,
                noise_max=noise_max,
                sweep_points=sweep_points,
                episodes_per_point=episodes_per_point,
                progress_callback=update_seed_progress,
            )
        )
    return aggregate_sweep_results(results)
