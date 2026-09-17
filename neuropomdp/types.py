"""Shared types, enums, and result dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np


Array: TypeAlias = jax.Array
JsonScalar: TypeAlias = str | int | float | bool | None
Serializable: TypeAlias = JsonScalar | list["Serializable"] | dict[str, "Serializable"]


class StateID(IntEnum):
    """Hidden-state indices for the benchmark environment."""

    LEFT_SECRET = 0
    RIGHT_SECRET = 1
    LEFT_REVEALED = 2
    RIGHT_REVEALED = 3
    LEFT_SUCCESS = 4
    LEFT_FAILURE = 5
    RIGHT_SUCCESS = 6
    RIGHT_FAILURE = 7


class ObservationID(IntEnum):
    """Observation indices for the benchmark environment."""

    AMBIGUOUS = 0
    LEFT_CUE = 1
    RIGHT_CUE = 2
    SUCCESS = 3
    FAILURE = 4


class ActionID(IntEnum):
    """Action indices for the benchmark environment."""

    INSPECT = 0
    EXPLOIT_LEFT = 1
    EXPLOIT_RIGHT = 2


@dataclass(slots=True)
class InferenceResult:
    """Exact Bayesian inference diagnostics."""

    prior: Array
    posterior: Array
    free_energy: Array
    kl_divergence: Array
    expected_log_likelihood: Array
    prior_entropy: Array
    posterior_entropy: Array


@dataclass(slots=True)
class ActionDiagnostics:
    """Expected Free Energy diagnostics for each available action."""

    predicted_states: Array
    predicted_observations: Array
    pragmatic_costs: Array
    epistemic_values: Array
    efe_values: Array
    action_probabilities: Array


@dataclass(slots=True)
class EpisodeResult:
    """Per-episode trajectory and diagnostic output."""

    observations: list[int]
    actions: list[int]
    rewards: list[float]
    hidden_states: list[int]
    beliefs: list[list[float]]
    posterior_entropies: list[float]
    free_energies: list[float]
    pragmatic_values: list[list[float]]
    epistemic_values: list[list[float]]
    efe_values: list[list[float]]
    action_probabilities: list[list[float]]
    selected_action_efe: list[float]
    total_reward: float
    success: bool
    mean_information_gain: float
    mean_pragmatic_cost: float


@dataclass(slots=True)
class AgentSummary:
    """Aggregated metrics for one agent across many episodes."""

    agent_name: str
    episode_count: int
    mean_reward: float
    reward_std: float
    reward_ci95_low: float
    reward_ci95_high: float
    success_rate: float
    success_ci95_low: float
    success_ci95_high: float
    mean_posterior_entropy: float
    mean_information_gain: float
    info_action_frequency: float
    action_counts: dict[str, int]


@dataclass(slots=True)
class BenchmarkResult:
    """Structured output of a benchmark run."""

    seed: int
    config: object
    agent_names: list[str]
    summaries: list[AgentSummary]
    episodes_by_agent: dict[str, list[EpisodeResult]]


@dataclass(slots=True)
class MultiSeedBenchmarkResult:
    """Aggregate statistics across multiple independent benchmark seeds."""

    seed_count: int
    seeds: list[int]
    config: object
    agent_names: list[str]
    summaries: list[AgentSummary]
    per_seed_results: list[BenchmarkResult]


AggregateBenchmarkResult = MultiSeedBenchmarkResult


@dataclass(slots=True)
class SweepResult:
    """Structured output of a parameter sweep."""

    seed: int
    config: object
    noise_values: list[float]
    agent_names: list[str]
    mean_reward: list[list[float]]
    success_rate: list[list[float]]
    mean_entropy: list[list[float]]
    info_action_frequency: list[list[float]]


@dataclass(slots=True)
class MultiSeedSweepResult:
    """Aggregate statistics across multiple sweep runs."""

    seed_count: int
    seeds: list[int]
    config: object
    noise_values: list[float]
    agent_names: list[str]
    mean_reward: list[list[float]]
    success_rate: list[list[float]]
    mean_entropy: list[list[float]]
    info_action_frequency: list[list[float]]
    per_seed_results: list[SweepResult]


AggregateSweepResult = MultiSeedSweepResult


def to_serializable(value: object) -> Serializable:
    """Convert nested dataclasses and arrays into JSON-friendly values."""

    if is_dataclass(value):
        return {key: to_serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (jnp.ndarray, np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if hasattr(value, "value"):
        return getattr(value, "value")
    raise TypeError(f"Unsupported value for serialization: {type(value)!r}")
