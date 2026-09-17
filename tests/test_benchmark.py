from __future__ import annotations

from dataclasses import replace

import numpy as np

from neuropomdp.benchmark import (
    aggregate_benchmark_results,
    benchmark_agents,
    benchmark_agents_multi_seed,
)
from neuropomdp.config import BenchmarkConfig
from neuropomdp.main import main
from neuropomdp.experiments import (
    aggregate_sweep_results,
    run_ablation_study,
    run_ablation_study_multi_seed,
    run_parameter_sweep,
    run_parameter_sweep_multi_seed,
    summarize_ablation_effect,
    summarize_sweep_effect,
)
from neuropomdp.visualization import plot_parameter_sweep


def test_benchmark_and_experiments_return_structured_results() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, seed=7, episodes=4, enable_plots=False),
        env=replace(base.env, episode_length=2),
    )

    benchmark = benchmark_agents(config)
    assert set(benchmark.agent_names) == {
        "random",
        "pragmatic_only",
        "active_inference",
        "bayesian_utility",
    }
    assert len(benchmark.summaries) == 4

    for summary in benchmark.summaries:
        assert summary.episode_count == 4
        assert np.isfinite(summary.mean_reward)
        assert summary.reward_ci95_low <= summary.mean_reward <= summary.reward_ci95_high
        assert np.isfinite(summary.mean_posterior_entropy)
        assert 0.0 <= summary.success_rate <= 1.0
        assert 0.0 <= summary.success_ci95_low <= summary.success_rate <= summary.success_ci95_high <= 1.0
        assert 0.0 <= summary.info_action_frequency <= 1.0

    ablation = run_ablation_study(config)
    assert set(ablation.agent_names) == {"active_inference", "no_epistemic"}
    assert len(ablation.summaries) == 2

    sweep = run_parameter_sweep(config)
    assert len(sweep.noise_values) == len(base.sweep_noise_values)
    assert len(sweep.mean_reward) == len(base.sweep_noise_values)
    assert len(sweep.mean_reward[0]) == 2
    assert len(sweep.success_rate[0]) == 2


def test_multi_seed_benchmark_runs_each_unique_seed() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, episodes=1, enable_plots=False),
    )

    results = benchmark_agents_multi_seed(config, [3, 3, 8])

    assert [result.seed for result in results] == [3, 8]
    assert all(result.agent_names for result in results)


def test_aggregate_multi_seed_results_exposes_seed_statistics() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, seed=7, episodes=2, enable_plots=False),
    )

    results = benchmark_agents_multi_seed(config, [0, 1])
    aggregated = aggregate_benchmark_results(results)

    assert aggregated.seed_count == 2
    assert aggregated.agent_names == results[0].agent_names
    assert len(aggregated.summaries) == len(results[0].summaries)
    for summary in aggregated.summaries:
        assert np.isfinite(summary.mean_reward)
        assert summary.reward_ci95_low <= summary.mean_reward <= summary.reward_ci95_high
        assert summary.success_ci95_low <= summary.success_rate <= summary.success_ci95_high


def test_multi_seed_experiments_are_supported() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, episodes=2, enable_plots=False),
    )

    ablation_results = run_ablation_study_multi_seed(config, [2, 3])
    sweep_results = run_parameter_sweep_multi_seed(config, [2, 3])

    assert len(ablation_results) == 2
    assert len(sweep_results) == 2
    assert all(result.agent_names == ["active_inference", "no_epistemic"] for result in ablation_results)
    assert all(len(result.noise_values) == len(base.sweep_noise_values) for result in sweep_results)


def test_aggregate_multi_seed_sweep_reports_seed_uncertainty() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, seed=7, episodes=2, enable_plots=False),
    )

    sweep_results = run_parameter_sweep_multi_seed(config, [0, 1])
    aggregated = aggregate_sweep_results(sweep_results)

    assert aggregated.seed_count == 2
    assert aggregated.noise_values == sweep_results[0].noise_values
    assert aggregated.agent_names == sweep_results[0].agent_names
    assert len(aggregated.mean_reward) == len(sweep_results[0].noise_values)
    assert len(aggregated.mean_reward[0]) == len(sweep_results[0].agent_names)


def test_summarize_ablation_effect_reports_seed_level_delta() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, seed=7, episodes=2, enable_plots=False),
    )

    ablation_results = run_ablation_study_multi_seed(config, [0, 1])
    effect = summarize_ablation_effect(ablation_results)

    assert effect["seed_count"] == 2
    assert "reward_delta" in effect
    assert "success_delta" in effect
    assert effect["active_agent"] == "active_inference"
    assert effect["baseline_agent"] == "no_epistemic"
    assert effect["reward_delta_ci95_low"] <= effect["reward_delta"] <= effect["reward_delta_ci95_high"]


def test_summarize_sweep_effect_reports_noise_level_benefit() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, seed=7, episodes=2, enable_plots=False),
    )

    sweep_results = run_parameter_sweep_multi_seed(config, [0, 1])
    effect = summarize_sweep_effect(sweep_results)

    assert effect["seed_count"] == 2
    assert effect["noise_values"] == sweep_results[0].noise_values
    assert len(effect["reward_deltas"]) == len(sweep_results[0].noise_values)
    assert len(effect["success_deltas"]) == len(sweep_results[0].noise_values)
    assert len(effect["reward_delta_ci95_low"]) == len(sweep_results[0].noise_values)
    assert len(effect["reward_delta_ci95_high"]) == len(sweep_results[0].noise_values)
    assert set(effect["reliably_reward_better_noise_values"]).issubset(sweep_results[0].noise_values)
    assert set(effect["reliably_success_better_noise_values"]).issubset(sweep_results[0].noise_values)


def test_multi_seed_sweep_plot_includes_error_bars() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, episodes=2, enable_plots=False),
    )

    sweep_results = run_parameter_sweep_multi_seed(config, [0, 1])
    aggregated = aggregate_sweep_results(sweep_results)
    figure = plot_parameter_sweep(aggregated)
    axis = figure.axes[0]

    assert len(axis.collections) > 0


def test_cli_multi_seed_sweep_writes_effect_summary(tmp_path) -> None:
    output_dir = tmp_path / "sweep"

    assert main(
        [
            "sweep",
            "--seeds",
            "0",
            "1",
            "--episodes",
            "1",
            "--sweep-noise-values",
            "0.05",
            "0.1",
            "--output-dir",
            str(output_dir),
            "--no-plots",
        ]
    ) == 0

    effect_path = output_dir / "sweep_multi_effect_summary.json"
    assert effect_path.exists()
    effect = effect_path.read_text(encoding="utf-8")
    assert "reliably_reward_better_noise_values" in effect

