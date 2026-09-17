from __future__ import annotations

import numpy as np
import pytest

from neuropomdp.config import ModelConfig
from neuropomdp.model import build_generative_model, validate_model


def test_model_dimensions_and_normalization() -> None:
    config = ModelConfig()
    model = build_generative_model(config)

    assert model.A.shape == (config.num_observations, config.num_states)
    assert model.B.shape == (config.num_actions, config.num_states, config.num_states)
    assert model.C.shape == (config.num_observations,)
    assert model.D.shape == (config.num_states,)

    assert np.allclose(np.asarray(model.A).sum(axis=0), 1.0)
    assert np.allclose(np.asarray(model.B).sum(axis=1), 1.0)
    assert np.isclose(float(np.asarray(model.C).sum()), 1.0)
    assert np.isclose(float(np.asarray(model.D).sum()), 1.0)


def test_validate_model_rejects_invalid_shapes_and_norms() -> None:
    config = ModelConfig()
    model = build_generative_model(config)
    A = np.asarray(model.A).copy()
    A[0, 0] = 2.0

    with pytest.raises(ValueError):
        validate_model(A, model.B, model.C, model.D, config)

    with pytest.raises(ValueError):
        validate_model(model.A[:-1], model.B, model.C, model.D, config)

