from __future__ import annotations

import inspect

import jax
import numpy as np

from neuropomdp.agent import ActiveInferenceAgent
from neuropomdp.config import EnvironmentConfig
from neuropomdp.environment import NeuroPOMDPEnvironment
from neuropomdp.types import ActionID


def test_environment_returns_valid_observations_and_transitions() -> None:
    env = NeuroPOMDPEnvironment(EnvironmentConfig())
    observation, info = env.reset(jax.random.PRNGKey(0))

    assert 0 <= observation < env.model.A.shape[0]
    assert "hidden_state" in info

    next_observation, reward, done, info2 = env.step(int(ActionID.INSPECT))
    assert 0 <= next_observation < env.model.A.shape[0]
    assert np.isfinite(reward)
    assert "hidden_state" in info2
    assert isinstance(done, bool)


def test_hidden_state_is_not_exposed_through_agent_api() -> None:
    signature = inspect.signature(ActiveInferenceAgent.update)
    assert "hidden_state" not in signature.parameters
    assert list(signature.parameters) == ["self", "observation", "action"]

