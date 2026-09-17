# NeuroPOMDP

NeuroPOMDP is a compact research project for discrete Active Inference in a partially observable decision-making task.

The project is designed as a small, CPU-friendly benchmark for studying when epistemic information-seeking becomes beneficial compared with purely pragmatic action selection.

## What it does

- builds a discrete generative model with `A`, `B`, `C`, and `D`
- performs exact Bayesian inference for hidden states
- computes variational free energy and Expected Free Energy
- compares full Active Inference against a no-epistemic baseline
- runs benchmark, ablation, and parameter-sweep experiments
- exposes the workflow through a local dashboard and CLI tooling

## Why this project exists

In ambiguous environments, a purely pragmatic policy may choose the action that looks best immediately, even when a better long-term strategy is to gather information first. NeuroPOMDP is built to make that tradeoff measurable and inspectable.

## Project structure

```text
neuropomdp/
├── agent.py
├── baselines.py
├── benchmark.py
├── cli_app.py
├── config.py
├── dashboard/
├── environment.py
├── experiments.py
├── inference.py
├── launcher.py
├── main.py
├── model.py
├── numerics.py
├── simulation.py
├── types.py
└── visualization.py

tests/
├── test_agent.py
├── test_benchmark.py
├── test_dashboard.py
├── test_dashboard_export.py
├── test_dashboard_history.py
├── test_efe.py
├── test_environment.py
├── test_inference.py
├── test_launcher.py
├── test_model.py
├── test_numerics.py
```

## Installation

```bash
python -m pip install -r requirements.txt
```

If you want to use the project in a local virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate  # Linux/macOS
# or .\.venv\Scripts\Activate.ps1  # Windows
python -m pip install -r requirements.txt
```

## Quick start

### Run the dashboard

```bash
python -m neuropomdp.launcher
```

### Run the demo

```bash
python -m neuropomdp.main demo
```

### Run the benchmark

```bash
python -m neuropomdp.main benchmark
```

### Run the ablation

```bash
python -m neuropomdp.main ablation
```

### Run the observation-noise sweep

```bash
python -m neuropomdp.main sweep
```

### Run tests

```bash
python -m pytest tests -q
```

## Dashboard

The project includes a Streamlit dashboard for exploring the model and results interactively.

It supports:

- Demo mode for single episodes
- Benchmark comparisons across agents
- Ablation analysis with and without epistemic value
- Observation-noise sweep analysis
- Research, history, and instructions tabs
- Downloadable experiment results and figures

## Research workflow

The benchmark reports include:

- cumulative reward
- success rate
- belief dynamics
- epistemic and pragmatic value decomposition
- confidence intervals across seeds
- reliable-advantage checks

For multi-seed studies, run the benchmark or sweep commands with explicit seeds to compare effects across repeated runs.

## Windows release

A packaged Windows build is provided in the release assets. Extract the archive and run `NeuroPOMDP.exe` from the extracted folder. The program starts the local dashboard and opens it in the default browser.

## Validation

The project is actively validated with automated tests and a packaged runtime smoke test. A fresh release build has been verified to start the dashboard runtime and resolve the required dependencies inside the frozen application bundle.

## License

This project is distributed under the repository's current licensing terms. See the project metadata and release notes for the exact policy for the version you are using.

## Notes

NeuroPOMDP is intended as an interpretable research benchmark rather than a broad production framework. The focus is on clarity, inspectability, and reproducible experimental comparison.

- `sweep_multi_summary.json`: mean sweep metrics aggregated by noise level and agent.
- `sweep_multi_effect_summary.json`: reward and success deltas, 95% CI bounds, and the noise
	values where the active agent has a reliable positive advantage.

For a quick local report with a smaller run:

```bash
python -m neuropomdp.main sweep \
	--seeds 0 1 2 \
	--episodes 32 \
	--sweep-noise-values 0.05 0.15 0.30 0.45 \
	--output-dir outputs/research_sweep \
	--no-plots
```

The dashboard exposes the same analysis interactively. Set `Sweep seeds` to a value greater
than one in the advanced settings, run the parameter sweep, and inspect the Research tab for
the per-noise effect table and confidence intervals. Use at least three seeds for exploratory
analysis and more seeds for a final report.

For the Streamlit dashboard directly:

```bash
python -m neuropomdp.launcher
```

## Reproducibility

All experiments accept:

- random seed,
- episode count,
- episode length,
- output directory,
- plotting toggle,
- environment parameters.

Using the same configuration and seed yields the same sampled episodes.

The multi-seed benchmark runs each seed independently and writes
`benchmark_multi.json`. Each seed reports mean reward, 95% reward confidence
intervals, and success rate so results can be compared without relying on a
single random initialization. Multi-seed ablation additionally writes an
aggregated `ablation_multi_summary.json`; multi-seed sweep writes the three
files described above, including the effect summary used to identify robust
noise regimes.
