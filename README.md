# NeuroPOMDP

NeuroPOMDP is a compact research project for discrete Active Inference in a partially observable decision-making task.

The project is designed as a small, CPU-friendly benchmark for studying when epistemic information-seeking can be beneficial compared with purely pragmatic action selection.

## Research question

In an ambiguous environment, when does gathering information before acting provide an advantage over choosing actions only according to their immediate pragmatic value?

NeuroPOMDP makes this trade-off measurable by comparing a full Active Inference agent with a baseline that does not use epistemic value.

The project focuses on:

* hidden-state inference;
* epistemic and pragmatic value;
* Expected Free Energy;
* observation noise;
* repeated experiments across random seeds;
* interpretable comparisons between agents.

## What it does

NeuroPOMDP:

* builds a discrete generative model with `A`, `B`, `C`, and `D`;
* performs exact Bayesian inference for hidden states;
* computes variational free energy and Expected Free Energy;
* compares full Active Inference against a no-epistemic baseline;
* runs benchmark, ablation, and parameter-sweep experiments;
* supports multi-seed experiments and confidence-interval analysis;
* provides a local Streamlit dashboard for inspecting experiments and results;
* provides CLI commands for running experiments directly;
* includes automated tests and a packaged Windows runtime.

## How it works

At a high level, the project follows this workflow:

```text
Environment
    ↓
Observations
    ↓
Bayesian inference
    ↓
Beliefs about hidden states
    ↓
Expected Free Energy
    ↓
Action selection
    ↓
Environment
```

The agent combines information-seeking and goal-directed considerations when selecting actions.

The baseline removes the epistemic component so that the effect of information-seeking can be examined experimentally.

More detailed descriptions of the architecture, model, and experiments are provided in the `docs/` directory.

## Project structure

```text
NeuroPOMDP/
├── neuropomdp/
│   ├── agent.py
│   ├── baselines.py
│   ├── benchmark.py
│   ├── config.py
│   ├── dashboard/
│   ├── environment.py
│   ├── experiments.py
│   ├── inference.py
│   ├── launcher.py
│   ├── main.py
│   ├── model.py
│   ├── numerics.py
│   ├── simulation.py
│   ├── types.py
│   └── visualization.py
│
├── tests/
│   ├── test_agent.py
│   ├── test_benchmark.py
│   ├── test_dashboard.py
│   ├── test_dashboard_export.py
│   ├── test_dashboard_history.py
│   ├── test_efe.py
│   ├── test_environment.py
│   ├── test_inference.py
│   ├── test_launcher.py
│   ├── test_model.py
│   └── test_numerics.py
│
├── docs/
├── examples/
├── packaging/
├── README.md
├── LICENSE
├── pytest.ini
└── requirements.txt
```

## Installation

Python dependencies can be installed with:

```bash
python -m pip install -r requirements.txt
```

For local development, the project can also be installed in editable mode:

```bash
python -m pip install -e .[dev]
```

Alternatively, install the development requirements file:

```bash
python -m pip install -r requirements-dev.txt
```

For an isolated local environment:

### Linux / macOS

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Quick start

### Run the dashboard

```bash
python -m neuropomdp.launcher
```

The launcher starts the local Streamlit dashboard.

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

The project includes a Streamlit dashboard for exploring the model and experiment results interactively.

It provides:

* demo mode for individual episodes;
* benchmark comparisons between agents;
* ablation analysis with and without epistemic value;
* observation-noise sweep analysis;
* research and history views;
* instructions and experiment information;
* downloadable experiment results and figures.

The dashboard is intended primarily as an inspection and analysis interface rather than as a separate implementation of the research model.

## Experiments

The project currently includes several experiment types.

### Benchmark

Compares agent performance using metrics such as:

* cumulative reward;
* success rate;
* confidence intervals across seeds.

### Ablation

Compares the full Active Inference agent with a baseline without the epistemic component.

This provides a direct way to examine the contribution of information-seeking behaviour.

### Observation-noise sweep

Changes the observation-noise level and measures how agent behaviour and performance change under different levels of uncertainty.

Example:

```bash
python -m neuropomdp.main sweep \
    --seeds 0 1 2 \
    --episodes 32 \
    --sweep-noise-values 0.05 0.15 0.30 0.45 \
    --output-dir outputs/research_sweep \
    --no-plots
```

For exploratory analysis, multiple seeds should be used rather than relying on a single random run.

Detailed experiment descriptions and parameters are documented in:

```text
docs/experiments.md
```

## Results

The project supports analysis of:

* cumulative reward;
* success rate;
* belief dynamics;
* epistemic value;
* pragmatic value;
* Expected Free Energy;
* confidence intervals across random seeds;
* changes in performance under observation noise.

Multi-seed experiments can be used to compare agents across repeated runs instead of relying on a single initialization.

Example research outputs may include:

```text
benchmark_multi.json
ablation_multi_summary.json
sweep_multi_summary.json
sweep_multi_effect_summary.json
```

The exact contents and interpretation of these files are described in the experiment documentation.

## Reproducibility

Experiments support explicit configuration of:

* random seeds;
* episode count;
* episode length;
* output directory;
* plotting;
* environment parameters.

Using the same configuration and seed is intended to reproduce the same sampled episodes.

Multi-seed experiments run each seed independently and aggregate the resulting metrics.

A typical reproducibility workflow is:

```text
1. Install the dependencies
2. Select the experiment and configuration
3. Specify one or more random seeds
4. Run the experiment
5. Save the generated results
6. Inspect the resulting metrics and figures
```

See:

```text
docs/reproducibility.md
```

for the currently supported reproducibility workflow.

## Documentation

Additional documentation is organized by topic:

```text
docs/
├── architecture.md
├── model.md
├── experiments.md
└── reproducibility.md
```

### Architecture

Describes the main components of the project and how data flows between the environment, inference, agent, experiments, and dashboard.

### Model

Describes the generative model, hidden states, observations, Bayesian inference, variational free energy, and Expected Free Energy as implemented in the project.

### Experiments

Documents the available experiments, parameters, metrics, seeds, and output files.

### Reproducibility

Documents the environment and commands needed to reproduce the supported experiments.

## Windows release

A packaged Windows build is provided in the release assets.

After extracting the release archive, run:

```text
NeuroPOMDP.exe
```

The launcher starts the local dashboard runtime.

The packaged application has also been validated with a runtime smoke test to check that the frozen application can start and serve the dashboard correctly.

## Validation

The project uses automated tests to validate the main components of the implementation.

Run the test suite with:

```bash
python -m pytest tests -q
```

The packaged Windows application is additionally checked with runtime smoke tests to verify that the frozen application can start its local server and resolve the required dependencies.

## Development

The project is intentionally kept relatively small.

NeuroPOMDP is intended as an interpretable research benchmark rather than a broad production framework. The focus is on:

* clear model implementation;
* inspectable inference;
* controlled experimental comparisons;
* reproducible configurations;
* understandable experiment outputs.

Changes to the research model should therefore be accompanied by corresponding tests and, where appropriate, updated experiment documentation.

## License

This project is distributed under the repository's current licensing terms.

See the repository license file for the exact terms applicable to the project.

## Citation

If you use NeuroPOMDP in research or an experiment, a citation file is provided in the repository when available:

```text
CITATION.cff
```

## Status

NeuroPOMDP is an ongoing research and experimental project.

The current version focuses on building a small and inspectable environment for studying Active Inference, epistemic information-seeking, and decision-making under partial observability.

The project is not intended to claim that Active Inference is universally superior to pragmatic decision-making. Its purpose is to provide a controlled setting in which the difference between these strategies can be measured and examined.
