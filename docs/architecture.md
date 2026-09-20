# Architecture

NeuroPOMDP is organized around a small discrete Active Inference workflow.

## Main Components

The core model is built in `neuropomdp.model`. It constructs the categorical `A`, `B`, `C`, and `D` arrays used by the environment, inference code, and agents.

The environment in `neuropomdp.environment` samples hidden states, observations, transitions, and rewards for the benchmark task.

Inference is implemented in `neuropomdp.inference` and updates beliefs over hidden states from observations using exact Bayesian inference.

Action evaluation is implemented in `neuropomdp.efe`. It computes pragmatic cost, epistemic value, expected free energy, and action probabilities.

Agents in `neuropomdp.agent` combine inference and action selection. The project includes an Active Inference agent, pragmatic-only baselines, and a random baseline.

Experiment orchestration lives in `neuropomdp.main`, `neuropomdp.benchmark`, and `neuropomdp.experiments`.

The Streamlit dashboard in `neuropomdp.dashboard` provides an interactive interface for running and inspecting experiments.

## Data Flow

```text
Configuration
    -> Generative model
    -> Environment
    -> Observation
    -> Bayesian inference
    -> Belief state
    -> Expected Free Energy
    -> Action selection
    -> Environment transition
    -> Metrics and visualizations
```

## Design Intent

The project favors clarity and reproducibility over broad framework generality. Most modules are intentionally small so that the model assumptions, inference steps, and experiment outputs remain inspectable.
