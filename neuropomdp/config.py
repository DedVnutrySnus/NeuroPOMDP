"""Configuration dataclasses for models, inference, simulation, and sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

from .numerics import configure_jax_precision


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration of the categorical generative model."""

    num_states: int = 8
    num_observations: int = 5
    num_actions: int = 3
    observation_noise: float = 0.15
    information_quality: float = 0.95
    transition_stochasticity: float = 0.05
    prior_left_prob: float = 0.5
    success_reward: float = 1.0
    failure_reward: float = -1.0
    info_reward: float = -0.2
    cue_reward: float = -0.05
    ambiguous_reward: float = -0.02
    preference_strength: float = 3.0
    dtype: str = "float32"
    enable_x64: bool = False


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Configuration for Bayesian inference."""

    eps: float = 1e-8


@dataclass(frozen=True, slots=True)
class ActionSelectionConfig:
    """Configuration for action selection."""

    precision: float = 8.0
    deterministic: bool = False
    use_epistemic: bool = True
    planning_horizon: int = 2
    planning_discount: float = 1.0


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Configuration for the benchmark POMDP environment."""

    model: ModelConfig = field(default_factory=ModelConfig)
    episode_length: int = 2
    terminate_on_terminal: bool = True


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuration for episode simulation."""

    seed: int = 0
    episodes: int = 64
    episode_length: int = 2
    enable_plots: bool = True
    output_dir: str = "outputs"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration for benchmark and sweep experiments."""

    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    selection: ActionSelectionConfig = field(default_factory=ActionSelectionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    sweep_noise_values: tuple[float, ...] = (0.05, 0.15, 0.3, 0.45)


def dtype_from_name(name: str) -> jnp.dtype:
    """Resolve a dtype name to a JAX dtype."""

    if name == "float64":
        return jnp.float64
    return jnp.float32


def apply_precision(config: ModelConfig) -> None:
    """Apply the global JAX precision setting."""

    configure_jax_precision(config.enable_x64)
