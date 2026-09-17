from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np

from neuropomdp.agent import ActiveInferenceAgent
from neuropomdp.config import ActionSelectionConfig, BenchmarkConfig, ModelConfig
from neuropomdp.model import build_generative_model
from neuropomdp.simulation import run_episode
from neuropomdp.types import ActionID


def test_same_seed_produces_same_episode() -> None:
    base = BenchmarkConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, seed=11, episodes=1, enable_plots=False),
        env=replace(base.env, episode_length=2),
    )
    model = build_generative_model(config.env.model)

    agent_one = ActiveInferenceAgent(model=model, inference_config=config.inference, selection_config=config.selection)
    episode_one = run_episode(agent_one, config.env, jax.random.PRNGKey(config.simulation.seed))

    agent_two = ActiveInferenceAgent(model=model, inference_config=config.inference, selection_config=config.selection)
    episode_two = run_episode(agent_two, config.env, jax.random.PRNGKey(config.simulation.seed))

    assert episode_one.actions == episode_two.actions
    assert episode_one.observations == episode_two.observations
    assert np.allclose(np.asarray(episode_one.rewards), np.asarray(episode_two.rewards))
    assert np.allclose(np.asarray(episode_one.beliefs), np.asarray(episode_two.beliefs))
    assert episode_one.total_reward == episode_two.total_reward


def test_informative_inspect_is_preferred_with_two_step_planning() -> None:
    model = build_generative_model(
        ModelConfig(
            observation_noise=0.3,
            information_quality=1.0,
            transition_stochasticity=0.0,
        )
    )
    agent = ActiveInferenceAgent(
        model=model,
        selection_config=ActionSelectionConfig(
            deterministic=True,
            planning_horizon=2,
        ),
    )

    action, diagnostics = agent.select_action()

    assert action == int(ActionID.INSPECT)
    assert float(diagnostics.efe_values[ActionID.INSPECT]) < float(
        diagnostics.efe_values[ActionID.EXPLOIT_LEFT]
    )

