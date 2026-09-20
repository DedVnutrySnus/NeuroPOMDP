"""NeuroPOMDP package.

This package implements a small discrete Active Inference research codebase
for a partially observable decision-making benchmark.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .agent import ActiveInferenceAgent
from .benchmark import benchmark_agents
from .config import (
    ActionSelectionConfig,
    BenchmarkConfig,
    EnvironmentConfig,
    InferenceConfig,
    ModelConfig,
    SimulationConfig,
    configure_jax_precision,
)
from .environment import NeuroPOMDPEnvironment
from .experiments import run_ablation_study, run_parameter_sweep
from .model import GenerativeModel, build_generative_model
from .simulation import run_episode, run_benchmark
from .types import (
    ActionID,
    AgentSummary,
    BenchmarkResult,
    EpisodeResult,
    InferenceResult,
    ObservationID,
    Serializable,
    StateID,
    SweepResult,
    to_serializable,
)

__all__ = [
    "ActiveInferenceAgent",
    "ActionID",
    "ActionSelectionConfig",
    "AgentSummary",
    "BenchmarkConfig",
    "BenchmarkResult",
    "EnvironmentConfig",
    "EpisodeResult",
    "GenerativeModel",
    "InferenceConfig",
    "InferenceResult",
    "ModelConfig",
    "NeuroPOMDPEnvironment",
    "ObservationID",
    "Serializable",
    "SimulationConfig",
    "StateID",
    "SweepResult",
    "__version__",
    "benchmark_agents",
    "build_generative_model",
    "configure_jax_precision",
    "run_ablation_study",
    "run_benchmark",
    "run_episode",
    "run_parameter_sweep",
    "to_serializable",
]
