"""Categorical generative model for the NeuroPOMDP benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelConfig, apply_precision, dtype_from_name
from .numerics import normalize
from .types import ActionID, ObservationID, StateID


@dataclass(slots=True)
class GenerativeModel:
    """Discrete generative model with A, B, C, and D matrices."""

    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray
    D: jnp.ndarray
    config: ModelConfig

    @property
    def reward_vector(self) -> jnp.ndarray:
        """Return rewards associated with the model's hidden states."""

        return jnp.asarray(
            [
                0.0,
                0.0,
                self.config.info_reward,
                self.config.info_reward,
                self.config.success_reward,
                self.config.failure_reward,
                self.config.success_reward,
                self.config.failure_reward,
            ],
            dtype=self.A.dtype,
        )

    def validate(self) -> None:
        """Validate shapes and normalization."""

        validate_model(self.A, self.B, self.C, self.D, self.config)

    def predict_next_belief(self, belief: jnp.ndarray, action: int) -> jnp.ndarray:
        """Predict the next state belief under one action."""

        predicted = self.B[action] @ belief
        return normalize(predicted)

    def predict_observation(self, state_belief: jnp.ndarray) -> jnp.ndarray:
        """Predict observations from a state belief."""

        return normalize(self.A @ state_belief)


def _make_observation_model(config: ModelConfig, dtype: jnp.dtype) -> jnp.ndarray:
    """Construct the likelihood matrix ``A[o, s]``."""

    num_obs = config.num_observations
    num_states = config.num_states
    A = np.zeros((num_obs, num_states), dtype=np.float64)
    noise = float(config.observation_noise)
    cue_mass = noise / 2.0
    secret_mass = 1.0 - noise
    revealed_mass = 1.0 - noise
    terminal_mass = 1.0 - noise

    secret_columns = (StateID.LEFT_SECRET, StateID.RIGHT_SECRET)
    for state in secret_columns:
        A[ObservationID.AMBIGUOUS, state] = secret_mass
        A[ObservationID.LEFT_CUE, state] = cue_mass
        A[ObservationID.RIGHT_CUE, state] = cue_mass

    A[ObservationID.LEFT_CUE, StateID.LEFT_REVEALED] = revealed_mass
    A[ObservationID.AMBIGUOUS, StateID.LEFT_REVEALED] = noise

    A[ObservationID.RIGHT_CUE, StateID.RIGHT_REVEALED] = revealed_mass
    A[ObservationID.AMBIGUOUS, StateID.RIGHT_REVEALED] = noise

    A[ObservationID.SUCCESS, StateID.LEFT_SUCCESS] = terminal_mass
    A[ObservationID.AMBIGUOUS, StateID.LEFT_SUCCESS] = noise

    A[ObservationID.FAILURE, StateID.LEFT_FAILURE] = terminal_mass
    A[ObservationID.AMBIGUOUS, StateID.LEFT_FAILURE] = noise

    A[ObservationID.SUCCESS, StateID.RIGHT_SUCCESS] = terminal_mass
    A[ObservationID.AMBIGUOUS, StateID.RIGHT_SUCCESS] = noise

    A[ObservationID.FAILURE, StateID.RIGHT_FAILURE] = terminal_mass
    A[ObservationID.AMBIGUOUS, StateID.RIGHT_FAILURE] = noise

    return jnp.asarray(A, dtype=dtype)


def _make_transition_model(config: ModelConfig, dtype: jnp.dtype) -> jnp.ndarray:
    """Construct the transition tensor ``B[a, s_next, s_prev]``."""

    S = config.num_states
    A = config.num_actions
    B = np.zeros((A, S, S), dtype=np.float64)
    slip = float(config.transition_stochasticity)
    reveal = float(config.information_quality)

    # Inspect action.
    for prev in range(S):
        if prev == int(StateID.LEFT_SECRET):
            B[ActionID.INSPECT, StateID.LEFT_REVEALED, prev] = reveal
            B[ActionID.INSPECT, StateID.LEFT_SECRET, prev] = 1.0 - reveal
        elif prev == int(StateID.RIGHT_SECRET):
            B[ActionID.INSPECT, StateID.RIGHT_REVEALED, prev] = reveal
            B[ActionID.INSPECT, StateID.RIGHT_SECRET, prev] = 1.0 - reveal
        else:
            B[ActionID.INSPECT, prev, prev] = 1.0

    # Exploit left.
    for prev in range(S):
        if prev in (int(StateID.LEFT_SECRET), int(StateID.LEFT_REVEALED)):
            B[ActionID.EXPLOIT_LEFT, StateID.LEFT_SUCCESS, prev] = 1.0 - slip
            B[ActionID.EXPLOIT_LEFT, StateID.LEFT_FAILURE, prev] = slip
        elif prev in (int(StateID.RIGHT_SECRET), int(StateID.RIGHT_REVEALED)):
            B[ActionID.EXPLOIT_LEFT, StateID.LEFT_FAILURE, prev] = 1.0 - slip
            B[ActionID.EXPLOIT_LEFT, StateID.LEFT_SUCCESS, prev] = slip
        else:
            B[ActionID.EXPLOIT_LEFT, prev, prev] = 1.0

    # Exploit right.
    for prev in range(S):
        if prev in (int(StateID.RIGHT_SECRET), int(StateID.RIGHT_REVEALED)):
            B[ActionID.EXPLOIT_RIGHT, StateID.RIGHT_SUCCESS, prev] = 1.0 - slip
            B[ActionID.EXPLOIT_RIGHT, StateID.RIGHT_FAILURE, prev] = slip
        elif prev in (int(StateID.LEFT_SECRET), int(StateID.LEFT_REVEALED)):
            B[ActionID.EXPLOIT_RIGHT, StateID.RIGHT_FAILURE, prev] = 1.0 - slip
            B[ActionID.EXPLOIT_RIGHT, StateID.RIGHT_SUCCESS, prev] = slip
        else:
            B[ActionID.EXPLOIT_RIGHT, prev, prev] = 1.0

    return jnp.asarray(B, dtype=dtype)


def _make_preferences(config: ModelConfig, dtype: jnp.dtype) -> jnp.ndarray:
    """Construct categorical preferences over observations.

    The preference convention is a categorical distribution over observations,
    not a raw utility vector. Pragmatic cost is computed as the cross-entropy
    between predicted outcomes and this distribution.
    """

    rewards = jnp.asarray(
        [
            config.ambiguous_reward,
            config.cue_reward,
            config.cue_reward,
            config.success_reward,
            config.failure_reward,
        ],
        dtype=dtype,
    )
    logits = config.preference_strength * rewards
    return jax.nn.softmax(logits).astype(dtype)


def _make_prior(config: ModelConfig, dtype: jnp.dtype) -> jnp.ndarray:
    """Construct the prior over hidden states."""

    D = np.zeros((config.num_states,), dtype=np.float64)
    left = float(config.prior_left_prob)
    D[StateID.LEFT_SECRET] = left
    D[StateID.RIGHT_SECRET] = 1.0 - left
    return jnp.asarray(D, dtype=dtype)


def build_generative_model(config: ModelConfig) -> GenerativeModel:
    """Build and validate the generative model."""

    if config.num_states != 8 or config.num_observations != 5 or config.num_actions != 3:
        raise ValueError("This benchmark expects 8 states, 5 observations, and 3 actions.")
    if not 0.0 <= config.observation_noise <= 1.0:
        raise ValueError("observation_noise must lie in [0, 1].")
    if not 0.0 <= config.transition_stochasticity <= 1.0:
        raise ValueError("transition_stochasticity must lie in [0, 1].")
    if not 0.0 <= config.information_quality <= 1.0:
        raise ValueError("information_quality must lie in [0, 1].")
    if not 0.0 <= config.prior_left_prob <= 1.0:
        raise ValueError("prior_left_prob must lie in [0, 1].")
    apply_precision(config)
    dtype = dtype_from_name(config.dtype)
    A = _make_observation_model(config, dtype)
    B = _make_transition_model(config, dtype)
    C = _make_preferences(config, dtype)
    D = _make_prior(config, dtype)
    model = GenerativeModel(A=A, B=B, C=C, D=D, config=config)
    model.validate()
    return model


def validate_model(
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    D: jnp.ndarray,
    config: ModelConfig,
    atol: float = 1e-5,
) -> None:
    """Validate matrix dimensions and categorical normalization."""

    if A.shape != (config.num_observations, config.num_states):
        raise ValueError(f"Invalid A shape: {A.shape}")
    if B.shape != (config.num_actions, config.num_states, config.num_states):
        raise ValueError(f"Invalid B shape: {B.shape}")
    if C.shape != (config.num_observations,):
        raise ValueError(f"Invalid C shape: {C.shape}")
    if D.shape != (config.num_states,):
        raise ValueError(f"Invalid D shape: {D.shape}")
    if jnp.any(A < 0) or jnp.any(B < 0) or jnp.any(C < 0) or jnp.any(D < 0):
        raise ValueError("Probability distributions must be non-negative.")
    if not np.allclose(np.asarray(A).sum(axis=0), 1.0, atol=atol):
        raise ValueError("Each column of A must sum to 1.")
    if not np.allclose(np.asarray(B).sum(axis=1), 1.0, atol=atol):
        raise ValueError("Each column of each B[a] must sum to 1.")
    if not np.isclose(float(np.asarray(C).sum()), 1.0, atol=atol):
        raise ValueError("C must sum to 1.")
    if not np.isclose(float(np.asarray(D).sum()), 1.0, atol=atol):
        raise ValueError("D must sum to 1.")
