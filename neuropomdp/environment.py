"""Custom POMDP environment for the NeuroPOMDP benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from .config import EnvironmentConfig
from .model import GenerativeModel, build_generative_model
from .numerics import categorical_sample
from .types import ObservationID, StateID


@dataclass(slots=True)
class NeuroPOMDPEnvironment:
    """Small discrete benchmark environment with hidden state and observations."""

    config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: GenerativeModel = field(init=False)
    key: jax.Array | None = field(default=None, init=False)
    state: int = field(default=0, init=False)
    step_count: int = field(default=0, init=False)
    done: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.model = build_generative_model(self.config.model)

    @property
    def reward_vector(self) -> jnp.ndarray:
        """Map hidden states to scalar reward for benchmarking."""

        return jnp.asarray(
            [
                0.0,
                0.0,
                self.config.model.info_reward,
                self.config.model.info_reward,
                self.config.model.success_reward,
                self.config.model.failure_reward,
                self.config.model.success_reward,
                self.config.model.failure_reward,
            ],
            dtype=self.model.A.dtype,
        )

    def reset(self, key: jax.Array) -> tuple[int, dict[str, int]]:
        """Reset the environment and return the initial observation."""

        self.key = key
        key_state, key_obs = jax.random.split(key)
        self.state = int(categorical_sample(key_state, self.model.D))
        observation = int(categorical_sample(key_obs, self.model.A[:, self.state]))
        self.step_count = 0
        self.done = False
        return observation, {"hidden_state": self.state, "step_count": self.step_count}

    def _sample_next_state(self, action: int) -> int:
        assert self.key is not None
        self.key, subkey = jax.random.split(self.key)
        return int(categorical_sample(subkey, self.model.B[action, :, self.state]))

    def _sample_observation(self, state: int) -> int:
        assert self.key is not None
        self.key, subkey = jax.random.split(self.key)
        return int(categorical_sample(subkey, self.model.A[:, state]))

    def step(self, action: int) -> tuple[int, float, bool, dict[str, int]]:
        """Advance the environment by one action."""

        if self.done:
            return (
                int(ObservationID.AMBIGUOUS),
                0.0,
                True,
                {"hidden_state": self.state, "step_count": self.step_count, "terminal": 1},
            )

        next_state = self._sample_next_state(action)
        observation = self._sample_observation(next_state)
        reward = float(self.reward_vector[next_state])
        self.state = next_state
        self.step_count += 1
        self.done = bool(
            self.config.terminate_on_terminal
            and next_state
            in (
                StateID.LEFT_SUCCESS,
                StateID.LEFT_FAILURE,
                StateID.RIGHT_SUCCESS,
                StateID.RIGHT_FAILURE,
            )
        ) or self.step_count >= self.config.episode_length
        return observation, reward, self.done, {"hidden_state": self.state, "step_count": self.step_count}
