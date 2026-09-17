"""Active Inference agent and belief-updating utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from .config import ActionSelectionConfig, InferenceConfig
from .efe import evaluate_actions
from .inference import exact_posterior, inference_diagnostics
from .model import GenerativeModel
from .numerics import categorical_sample, normalize
from .types import ActionDiagnostics, InferenceResult, StateID


@dataclass(slots=True)
class ActiveInferenceAgent:
    """Discrete Active Inference agent with exact Bayesian updates."""

    model: GenerativeModel
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    selection_config: ActionSelectionConfig = field(default_factory=ActionSelectionConfig)
    belief: jnp.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    @property
    def name(self) -> str:
        return "active_inference"

    def reset(self) -> None:
        """Reset the internal belief to the prior."""

        self.belief = jnp.asarray(self.model.D)

    def infer(self, observation: int, action: int | None = None) -> InferenceResult:
        """Update beliefs after observing an outcome."""

        prior = self.belief if action is None else normalize(self.model.B[action] @ self.belief)
        posterior = exact_posterior(self.model.A, prior, observation)
        self.belief = posterior
        return inference_diagnostics(prior, posterior, self.model.A[observation])

    def evaluate(self) -> ActionDiagnostics:
        """Evaluate the available actions under the current belief."""

        return evaluate_actions(
            self.belief,
            self.model.A,
            self.model.B,
            self.model.C,
            precision=self.selection_config.precision,
            use_epistemic=self.selection_config.use_epistemic,
            planning_horizon=self.selection_config.planning_horizon,
            planning_discount=self.selection_config.planning_discount,
            reward_vector=self.model.reward_vector,
            terminal_states=jnp.asarray(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype=self.model.A.dtype,
            ),
        )

    def select_action(self, key: jax.Array | None = None) -> tuple[int, ActionDiagnostics]:
        """Select an action according to the configured strategy."""

        diagnostics = self.evaluate()
        if self.selection_config.deterministic:
            action = int(jnp.argmin(diagnostics.efe_values))
            return action, diagnostics
        assert key is not None, "A PRNG key is required for stochastic action selection."
        action = int(categorical_sample(key, diagnostics.action_probabilities))
        return action, diagnostics

    def update(self, observation: int, action: int | None = None) -> InferenceResult:
        """Observe the environment and update the posterior belief."""

        return self.infer(observation, action=action)


@dataclass(slots=True)
class BayesianUtilityAgent(ActiveInferenceAgent):
    """Bayesian decision-maker without epistemic value."""

    def __post_init__(self) -> None:
        self.selection_config = ActionSelectionConfig(
            precision=self.selection_config.precision,
            deterministic=self.selection_config.deterministic,
            use_epistemic=False,
            planning_horizon=self.selection_config.planning_horizon,
            planning_discount=self.selection_config.planning_discount,
        )
        self.reset()

    @property
    def name(self) -> str:
        return "bayesian_utility"


@dataclass(slots=True)
class PragmaticOnlyAgent(BayesianUtilityAgent):
    """Alias for the pragmatic-only baseline."""

    @property
    def name(self) -> str:
        return "pragmatic_only"


@dataclass(slots=True)
class RandomAgent:
    """Uniform random baseline."""

    model: GenerativeModel
    belief: jnp.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    @property
    def name(self) -> str:
        return "random"

    def reset(self) -> None:
        """Reset the belief so inference metrics remain comparable."""

        self.belief = jnp.asarray(self.model.D)

    def infer(self, observation: int, action: int | None = None) -> InferenceResult:
        """Perform exact Bayesian inference even though actions are random."""

        prior = self.belief if action is None else normalize(self.model.B[action] @ self.belief)
        posterior = exact_posterior(self.model.A, prior, observation)
        self.belief = posterior
        return inference_diagnostics(prior, posterior, self.model.A[observation])

    def evaluate(self) -> ActionDiagnostics:
        """Provide a placeholder action diagnostic."""

        num_actions = self.model.B.shape[0]
        zeros = jnp.zeros((num_actions,), dtype=self.model.A.dtype)
        probs = jnp.ones((num_actions,), dtype=self.model.A.dtype) / float(num_actions)
        return ActionDiagnostics(
            predicted_states=jnp.zeros_like(self.model.B[:, :, 0]),
            predicted_observations=jnp.zeros((num_actions, self.model.A.shape[0]), dtype=self.model.A.dtype),
            pragmatic_costs=zeros,
            epistemic_values=zeros,
            efe_values=zeros,
            action_probabilities=probs,
        )

    def select_action(self, key: jax.Array | None = None) -> tuple[int, ActionDiagnostics]:
        """Sample uniformly at random."""

        assert key is not None, "A PRNG key is required."
        diagnostics = self.evaluate()
        action = int(jax.random.randint(key, (), 0, self.model.B.shape[0]))
        return action, diagnostics

    def update(self, observation: int, action: int | None = None) -> InferenceResult:
        """Update beliefs from observations."""

        return self.infer(observation, action=action)
