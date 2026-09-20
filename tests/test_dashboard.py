from __future__ import annotations

from dataclasses import replace

import numpy as np

from neuropomdp.config import BenchmarkConfig
from neuropomdp.dashboard import app as dashboard_app
from neuropomdp.dashboard.i18n import set_language, t
from neuropomdp.dashboard.formatting import (
    benchmark_rows,
    episode_rows,
    generate_action_explanation,
    rows_to_csv_text,
    rows_to_markdown,
    sweep_rows,
    sweep_effect_rows,
    extract_step_diagnostics,
)
from neuropomdp.dashboard.state import (
    build_dashboard_config,
    run_ablation_dashboard,
    run_benchmark_dashboard,
    run_demo,
    run_sweep_dashboard,
    run_sweep_dashboard_multi_seed,
)
from neuropomdp.experiments import aggregate_sweep_results, run_parameter_sweep_multi_seed
from neuropomdp.visualization import plot_benchmark_cumulative_reward


def test_dashboard_modules_import() -> None:
    assert hasattr(dashboard_app, "main")


def test_demo_step_index_skips_slider_for_single_action_episode(monkeypatch) -> None:
    def fail_slider(*args, **kwargs):
        raise AssertionError("slider should not be rendered")

    monkeypatch.setattr(dashboard_app.st, "slider", fail_slider)
    episode = type("Episode", (), {"actions": [0]})()

    assert dashboard_app._demo_step_index(episode) == 0


def test_dashboard_language_defaults_to_english() -> None:
    set_language("en")

    assert t("demo_heading") == "Demo"
    assert t("run_experiment") == "▶ Run Experiment"


def test_dashboard_language_switch_preserves_dashboard_api() -> None:
    set_language("ru")

    assert t("demo_heading") == "Демонстрация"
    assert t("run_experiment") == "▶ Запустить эксперимент"
    assert dashboard_app.observation_display(0) == "Неоднозначное"

    set_language("en")


def test_dashboard_instructions_tab_is_localized() -> None:
    set_language("en")
    assert t("tab_instructions") == "Instructions"

    set_language("ru")
    assert t("tab_instructions") == "Инструкция"

    set_language("en")


def test_dashboard_formatting_and_csv_serialization() -> None:
    base = BenchmarkConfig()
    config = build_dashboard_config(
        seed=3,
        episodes=2,
        episode_length=2,
        output_dir="outputs",
        show_plots=False,
        observation_noise=base.env.model.observation_noise,
        information_quality=base.env.model.information_quality,
        transition_stochasticity=base.env.model.transition_stochasticity,
        initial_uncertainty=base.env.model.prior_left_prob,
        precision=base.selection.precision,
        deterministic=base.selection.deterministic,
        inference_eps=base.inference.eps,
    )

    demo = run_demo(config)
    step = extract_step_diagnostics(demo, 0)
    explanation = generate_action_explanation(step)
    assert explanation
    assert "EFE" in rows_to_markdown(episode_rows(demo))

    benchmark = run_benchmark_dashboard(config)
    table_rows = benchmark_rows(benchmark)
    assert len(table_rows) == 3
    assert "Mean Reward" in rows_to_csv_text(table_rows)

    ablation = run_ablation_dashboard(config)
    assert len(benchmark_rows(ablation)) == 2

    sweep = run_sweep_dashboard(config, 0.05, 0.1, 2, 1)
    sweep_table = sweep_rows(sweep)
    assert len(sweep_table) == 6
    assert "Observation Noise" in rows_to_csv_text(sweep_table)


def test_dashboard_results_are_numeric_and_finite() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, seed=11, episodes=1, enable_plots=False),
        env=replace(base.env, episode_length=2),
    )

    benchmark = run_benchmark_dashboard(
        build_dashboard_config(
            seed=11,
            episodes=1,
            episode_length=2,
            output_dir="outputs",
            show_plots=False,
            observation_noise=config.env.model.observation_noise,
            information_quality=config.env.model.information_quality,
            transition_stochasticity=config.env.model.transition_stochasticity,
            initial_uncertainty=config.env.model.prior_left_prob,
            precision=config.selection.precision,
            deterministic=config.selection.deterministic,
            inference_eps=config.inference.eps,
        )
    )
    for summary in benchmark.summaries:
        assert np.isfinite(summary.mean_reward)
        assert np.isfinite(summary.mean_posterior_entropy)


def test_benchmark_cumulative_reward_plot_includes_error_bars() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, seed=7, episodes=4, enable_plots=False),
        env=replace(base.env, episode_length=2),
    )

    benchmark = run_benchmark_dashboard(config)
    fig = plot_benchmark_cumulative_reward(benchmark)
    ax = fig.axes[0]

    assert len(ax.collections) > 0 or any(hasattr(line, "has_xerr") and line.has_xerr for line in ax.lines)


def test_multi_seed_sweep_rows_include_reward_confidence_intervals() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, episodes=2, enable_plots=False),
    )

    aggregate = aggregate_sweep_results(run_parameter_sweep_multi_seed(config, [0, 1]))
    rows = sweep_rows(aggregate)

    assert rows
    assert "Reward CI 95% Low" in rows[0]
    assert rows[0]["Reward CI 95% Low"] <= rows[0]["Mean Reward"] <= rows[0]["Reward CI 95% High"]


def test_multi_seed_sweep_effect_rows_expose_reliable_advantage() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, episodes=2, enable_plots=False),
    )

    aggregate = aggregate_sweep_results(run_parameter_sweep_multi_seed(config, [0, 1]))
    rows = sweep_effect_rows(aggregate)

    assert len(rows) == len(aggregate.noise_values)
    assert "Reward Delta CI 95% Low" in rows[0]
    assert isinstance(rows[0]["Reliable Reward Advantage"], bool)


def test_dashboard_multi_seed_sweep_aggregates_unique_seeds() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, episodes=1, enable_plots=False),
    )

    result = run_sweep_dashboard_multi_seed(config, 0.05, 0.1, 2, 1, [3, 3, 8])

    assert result.seed_count == 2
    assert result.seeds == [3, 8]
    assert result.agent_names == ["random", "pragmatic_only", "active_inference"]


def test_dashboard_multi_seed_sweep_reward_chart_includes_error_bars() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, episodes=1, enable_plots=False),
    )

    result = run_sweep_dashboard_multi_seed(config, 0.05, 0.1, 2, 1, [3, 8])
    figure = dashboard_app._sweep_reward_figure(result)

    assert len(figure.axes[0].collections) > 0


def test_dashboard_multi_seed_sweep_success_chart_includes_error_bars() -> None:
    set_language("ru")
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, episodes=1, enable_plots=False),
    )

    result = run_sweep_dashboard_multi_seed(config, 0.05, 0.1, 2, 1, [3, 8])
    figure = dashboard_app._sweep_success_figure(result)

    assert len(figure.axes[0].collections) > 0
    assert figure.axes[0].get_ylim() == (0.0, 1.0)
    assert figure.axes[0].get_ylabel() == "Доля успеха"
    assert figure.axes[0].get_title() == "Доля успеха и шум наблюдений"
    set_language("en")
