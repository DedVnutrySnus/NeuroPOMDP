from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from neuropomdp.agent import ActiveInferenceAgent
from neuropomdp.config import ActionSelectionConfig, ModelConfig
from neuropomdp.efe import evaluate_actions
from neuropomdp.model import build_generative_model
from neuropomdp.types import ActionID


def test_state_and_observation_prediction_and_efe_decomposition() -> None:
    model = build_generative_model(ModelConfig())
    belief = jnp.asarray(model.D)
    diagnostics = evaluate_actions(
        belief,
        model.A,
        model.B,
        model.C,
        precision=8.0,
        use_epistemic=True,
    )

    assert np.isclose(float(diagnostics.action_probabilities.sum()), 1.0, atol=1e-6)
    assert np.all(np.asarray(diagnostics.epistemic_values) >= -1e-6)
    assert np.allclose(
        np.asarray(diagnostics.efe_values),
        np.asarray(diagnostics.pragmatic_costs) - np.asarray(diagnostics.epistemic_values),
        atol=1e-6,
    )

    for action in range(model.B.shape[0]):
        predicted_state = np.asarray(model.B[action] @ belief)
        predicted_observation = np.asarray(model.A @ predicted_state)
        assert np.allclose(np.asarray(diagnostics.predicted_states[action]), predicted_state, atol=1e-6)
        assert np.allclose(np.asarray(diagnostics.predicted_observations[action]), predicted_observation, atol=1e-6)


def test_action_selection_argmin_matches_efe() -> None:
    model = build_generative_model(ModelConfig())
    agent = ActiveInferenceAgent(
        model=model,
        selection_config=ActionSelectionConfig(precision=8.0, deterministic=True),
    )
    diagnostics = agent.evaluate()
    action, returned_diagnostics = agent.select_action()

    assert action == int(np.argmin(np.asarray(diagnostics.efe_values)))
    assert np.allclose(np.asarray(returned_diagnostics.efe_values), np.asarray(diagnostics.efe_values))

