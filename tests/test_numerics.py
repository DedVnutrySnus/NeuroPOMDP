from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from neuropomdp.numerics import categorical_sample, entropy, normalize, safe_log, softmax


def test_safe_log_and_normalize_are_stable() -> None:
    values = jnp.array([0.0, 1e-12, 1.0], dtype=jnp.float32)
    logged = safe_log(values)
    assert np.isfinite(np.asarray(logged)).all()

    normalized = normalize(jnp.array([1e-20, 1e-20], dtype=jnp.float32))
    assert np.isclose(float(normalized.sum()), 1.0)
    assert np.allclose(np.asarray(normalized), np.array([0.5, 0.5]), atol=1e-6)


def test_entropy_and_softmax_are_valid() -> None:
    uniform = jnp.array([0.5, 0.5], dtype=jnp.float32)
    assert np.isclose(float(entropy(uniform)), np.log(2.0), atol=1e-6)

    probs = softmax(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
    assert np.isclose(float(probs.sum()), 1.0)
    assert np.all(probs > 0.0)


def test_categorical_sample_is_reproducible() -> None:
    key = jax.random.PRNGKey(0)
    probs = jnp.array([0.1, 0.9], dtype=jnp.float32)
    first = int(categorical_sample(key, probs))
    second = int(categorical_sample(key, probs))
    assert first == second

