"""Expected Free Energy and action evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .numerics import entropy, normalize, safe_log, softmax
from .types import ActionDiagnostics


@jax.jit
def predict_state_for_action(belief: jnp.ndarray, transition_matrix: jnp.ndarray) -> jnp.ndarray:
    """Predict the next state belief under one action."""

    return normalize(transition_matrix @ belief)


@jax.jit
def predict_observation_for_state(state_belief: jnp.ndarray, likelihood: jnp.ndarray) -> jnp.ndarray:
    """Predict observations from a predicted state belief."""

    return normalize(likelihood @ state_belief)


@jax.jit
def pragmatic_cost(predicted_observation: jnp.ndarray, preferences: jnp.ndarray) -> jnp.ndarray:
    """Compute the expected preference cost.

    Lower values are better. Preferences are represented as a categorical
    distribution over observations.
    """

    return -jnp.sum(predicted_observation * safe_log(preferences))


@jax.jit
def epistemic_value(
    state_belief: jnp.ndarray,
    likelihood: jnp.ndarray,
    predicted_observation: jnp.ndarray,
) -> jnp.ndarray:
    """Compute expected information gain, i.e. mutual information."""

    observation_entropy = entropy(predicted_observation)
    conditional_entropy = jnp.sum(state_belief * entropy(likelihood, axis=0))
    return jnp.maximum(observation_entropy - conditional_entropy, 0.0)


@jax.jit
def efe_for_action(
    belief: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    likelihood: jnp.ndarray,
    preferences: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute pragmatic cost, epistemic value, and total EFE for one action."""

    predicted_state = predict_state_for_action(belief, transition_matrix)
    predicted_observation = predict_observation_for_state(predicted_state, likelihood)
    pragmatic = pragmatic_cost(predicted_observation, preferences)
    epistemic = epistemic_value(predicted_state, likelihood, predicted_observation)
    total = pragmatic - epistemic
    return predicted_state, predicted_observation, pragmatic, epistemic, total


@jax.jit
def action_probabilities(efe_values: jnp.ndarray, precision: float) -> jnp.ndarray:
    """Turn action EFE values into a categorical distribution."""

    return softmax(-precision * efe_values)


def _evaluate_one_step(
    belief: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    use_epistemic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate actions without looking beyond their immediate outcomes."""

    def _single_action(
        transition_matrix: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        predicted_state = predict_state_for_action(belief, transition_matrix)
        predicted_observation = predict_observation_for_state(predicted_state, A)
        pragmatic = pragmatic_cost(predicted_observation, C)
        epistemic = epistemic_value(predicted_state, A, predicted_observation)
        total = pragmatic - epistemic if use_epistemic else pragmatic
        return predicted_state, predicted_observation, pragmatic, epistemic, total

    return jax.vmap(_single_action)(B)


def _expected_next_cost(
    predicted_state: jnp.ndarray,
    predicted_observation: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    use_epistemic: bool,
) -> jnp.ndarray:
    """Estimate the value of the best action after the next observation."""

    next_costs = []
    for observation in range(A.shape[0]):
        posterior = normalize(predicted_state * A[observation])
        next_evaluation = _evaluate_one_step(posterior, A, B, C, use_epistemic)
        next_costs.append(jnp.min(next_evaluation[-1]))
    return jnp.sum(predicted_observation * jnp.stack(next_costs))


def _expected_next_reward(
    predicted_state: jnp.ndarray,
    predicted_observation: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    reward_vector: jnp.ndarray,
    terminal_states: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate the best non-terminal reward after the next observation."""

    continuation_mask = 1.0 - terminal_states
    next_rewards = []
    for observation in range(A.shape[0]):
        posterior = normalize(predicted_state * A[observation])
        continuation_belief = posterior * continuation_mask
        action_states = jax.vmap(lambda transition: transition @ continuation_belief)(B)
        next_rewards.append(jnp.max(action_states @ reward_vector))
    return jnp.sum(predicted_observation * jnp.stack(next_rewards))


def evaluate_actions(
    belief: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    precision: float,
    use_epistemic: bool = True,
    planning_horizon: int = 1,
    planning_discount: float = 1.0,
    reward_vector: jnp.ndarray | None = None,
    terminal_states: jnp.ndarray | None = None,
) -> ActionDiagnostics:
    """Evaluate all actions, optionally including one step of lookahead."""

    if planning_horizon < 1:
        raise ValueError("planning_horizon must be at least 1.")
    if not 0.0 <= planning_discount <= 1.0:
        raise ValueError("planning_discount must lie in [0, 1].")

    predicted_states, predicted_observations, pragmatic_costs, epistemic_values, efe_values = _evaluate_one_step(
        belief, A, B, C, use_epistemic
    )
    if planning_horizon > 1:
        if reward_vector is not None and terminal_states is not None:
            future_costs = -planning_discount * jnp.stack(
                [
                    _expected_next_reward(
                        predicted_states[action],
                        predicted_observations[action],
                        A,
                        B,
                        reward_vector,
                        terminal_states,
                    )
                    for action in range(B.shape[0])
                ]
            )
        else:
            future_costs = jnp.stack(
                [
                    _expected_next_cost(
                        predicted_states[action],
                        predicted_observations[action],
                        A,
                        B,
                        C,
                        use_epistemic,
                    )
                    for action in range(B.shape[0])
                ]
            )
        efe_values = efe_values + future_costs
    action_probs = action_probabilities(efe_values, precision)
    return ActionDiagnostics(
        predicted_states=predicted_states,
        predicted_observations=predicted_observations,
        pragmatic_costs=pragmatic_costs,
        epistemic_values=epistemic_values,
        efe_values=efe_values,
        action_probabilities=action_probs,
    )
