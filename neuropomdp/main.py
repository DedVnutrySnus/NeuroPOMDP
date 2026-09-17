"""Command line entry point for NeuroPOMDP experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax

from .agent import ActiveInferenceAgent
from .benchmark import aggregate_benchmark_results, benchmark_agents, benchmark_agents_multi_seed
from .config import BenchmarkConfig
from .experiments import (
    aggregate_sweep_results,
    run_ablation_study,
    run_ablation_study_multi_seed,
    run_parameter_sweep,
    run_parameter_sweep_multi_seed,
    summarize_sweep_effect,
)
from .model import build_generative_model
from .simulation import run_episode
from .types import to_serializable
from .visualization import (
    plot_ablation,
    plot_actions_over_time,
    plot_benchmark_cumulative_reward,
    plot_efe_values,
    plot_parameter_sweep,
    plot_posterior_entropy,
    plot_pragmatic_vs_epistemic,
    plot_state_and_belief,
)


def _build_config(args: argparse.Namespace) -> BenchmarkConfig:
    base = BenchmarkConfig()
    model = replace(
        base.env.model,
        observation_noise=args.observation_noise,
        information_quality=args.information_quality,
        transition_stochasticity=args.transition_stochasticity,
        prior_left_prob=args.prior_left_prob,
        dtype="float64" if args.float64 else base.env.model.dtype,
        enable_x64=bool(args.float64),
    )
    env = replace(base.env, model=model, episode_length=args.episode_length)
    simulation = replace(
        base.simulation,
        seed=args.seed,
        episodes=args.episodes,
        episode_length=args.episode_length,
        enable_plots=not args.no_plots,
        output_dir=args.output_dir,
    )
    selection = replace(
        base.selection,
        precision=args.precision,
        deterministic=args.deterministic,
    )
    return BenchmarkConfig(
        env=env,
        inference=base.inference,
        selection=selection,
        simulation=simulation,
        sweep_noise_values=tuple(args.sweep_noise_values),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(value), handle, indent=2, sort_keys=True)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--episode-length", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--observation-noise", type=float, default=0.15)
    parser.add_argument("--information-quality", type=float, default=0.95)
    parser.add_argument("--transition-stochasticity", type=float, default=0.05)
    parser.add_argument("--prior-left-prob", type=float, default=0.5)
    parser.add_argument("--precision", type=float, default=8.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--sweep-noise-values",
        type=float,
        nargs="+",
        default=[0.05, 0.15, 0.3, 0.45],
    )


def _run_demo(config: BenchmarkConfig) -> None:
    model = build_generative_model(config.env.model)
    agent = ActiveInferenceAgent(
        model=model,
        inference_config=config.inference,
        selection_config=config.selection,
    )
    episode = run_episode(agent, config.env, jax.random.PRNGKey(config.simulation.seed))
    output_dir = Path(config.simulation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "demo_episode.json", episode)
    if config.simulation.enable_plots:
        plot_state_and_belief(episode, output_dir / "figure_1_belief.png")
        plot_posterior_entropy(episode, output_dir / "figure_2_entropy.png")
        plot_actions_over_time(episode, output_dir / "figure_3_actions.png")
        plot_pragmatic_vs_epistemic(episode, output_dir / "figure_4_values.png")
        plot_efe_values(episode, output_dir / "figure_5_efe.png")
    print(f"Demo total reward: {episode.total_reward:.3f}")


def _run_benchmark(config: BenchmarkConfig, seeds: list[int] | None = None) -> None:
    if seeds:
        results = benchmark_agents_multi_seed(config, seeds)
        output_dir = Path(config.simulation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "benchmark_multi.json", results)
        for result in results:
            print(f"Seed {result.seed}:")
            for summary in result.summaries:
                print(
                    f"  {summary.agent_name}: reward={summary.mean_reward:.3f} "
                    f"[{summary.reward_ci95_low:.3f}, {summary.reward_ci95_high:.3f}], "
                    f"success={summary.success_rate:.3f}"
                )
        return
    result = benchmark_agents(config)
    output_dir = Path(config.simulation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "benchmark.json", result)
    if config.simulation.enable_plots:
        plot_benchmark_cumulative_reward(result, output_dir / "figure_6_benchmark_reward.png")
    for summary in result.summaries:
        print(
            f"{summary.agent_name}: reward={summary.mean_reward:.3f}, "
            f"success={summary.success_rate:.3f}, entropy={summary.mean_posterior_entropy:.3f}"
        )


def _run_ablation(config: BenchmarkConfig, seeds: list[int] | None = None) -> None:
    if seeds:
        results = run_ablation_study_multi_seed(config, seeds)
        output_dir = Path(config.simulation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "ablation_multi.json", results)
        aggregate = aggregate_benchmark_results(results)
        _write_json(output_dir / "ablation_multi_summary.json", aggregate)
        for result in results:
            print(f"Seed {result.seed}:")
            for summary in result.summaries:
                print(
                    f"  {summary.agent_name}: reward={summary.mean_reward:.3f}, "
                    f"success={summary.success_rate:.3f}"
                )
        return
    result = run_ablation_study(config)
    output_dir = Path(config.simulation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "ablation.json", result)
    if config.simulation.enable_plots:
        plot_ablation(result, output_dir / "figure_7_ablation.png")
    for summary in result.summaries:
        print(
            f"{summary.agent_name}: reward={summary.mean_reward:.3f}, "
            f"success={summary.success_rate:.3f}, entropy={summary.mean_posterior_entropy:.3f}"
        )


def _run_sweep(config: BenchmarkConfig, seeds: list[int] | None = None) -> None:
    if seeds:
        results = run_parameter_sweep_multi_seed(config, seeds)
        aggregate = aggregate_sweep_results(results)
        output_dir = Path(config.simulation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "sweep_multi.json", results)
        _write_json(output_dir / "sweep_multi_summary.json", aggregate)
        _write_json(output_dir / "sweep_multi_effect_summary.json", summarize_sweep_effect(results))
        for result in results:
            print(f"Seed {result.seed}: sweep completed for {len(result.noise_values)} noise settings.")
        effect = summarize_sweep_effect(results)
        reliable_noise = effect["reliably_reward_better_noise_values"]
        print(f"Reliable reward advantage noise values: {reliable_noise}")
        return
    result = run_parameter_sweep(config)
    output_dir = Path(config.simulation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "sweep.json", result)
    if config.simulation.enable_plots:
        plot_parameter_sweep(result, output_dir / "figure_8_sweep.png")
    print(f"Sweep completed for {len(result.noise_values)} noise settings.")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(prog="neuropomdp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo")
    benchmark = subparsers.add_parser("benchmark")
    ablation = subparsers.add_parser("ablation")
    sweep = subparsers.add_parser("sweep")
    for subparser in (demo, benchmark, ablation, sweep):
        _add_common_arguments(subparser)
    benchmark.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run independent benchmark repetitions for these seeds.",
    )
    ablation.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run independent ablation repetitions for these seeds.",
    )
    sweep.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run independent sweep repetitions for these seeds.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    config = _build_config(args)
    if config.env.model.enable_x64:
        jax.config.update("jax_enable_x64", True)

    if args.command == "demo":
        _run_demo(config)
    elif args.command == "benchmark":
        _run_benchmark(config, seeds=args.seeds)
    elif args.command == "ablation":
        _run_ablation(config, seeds=args.seeds)
    elif args.command == "sweep":
        _run_sweep(config, seeds=args.seeds)
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
