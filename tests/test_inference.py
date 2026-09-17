from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from neuropomdp.inference import exact_posterior, variational_free_energy
from neuropomdp.numerics import kl_divergence, normalize, safe_log


def test_exact_posterior_matches_bayes_rule() -> None:
    A = jnp.array(
        [
            [0.9, 0.2],
            [0.1, 0.8],
        ],
        dtype=jnp.float32,
    )
    D = jnp.array([0.6, 0.4], dtype=jnp.float32)
    observation = 0

    posterior = exact_posterior(A, D, observation)
    expected = normalize(A[observation] * D)
    assert np.allclose(np.asarray(posterior), np.asarray(expected), atol=1e-6)


def test_variational_free_energy_matches_identity() -> None:
    A = jnp.array(
        [
            [0.9, 0.2],
            [0.1, 0.8],
        ],
        dtype=jnp.float32,
    )
    D = jnp.array([0.6, 0.4], dtype=jnp.float32)
    observation = 0
    q = exact_posterior(A, D, observation)

    free_energy = variational_free_energy(q, D, A[observation])
    expected = kl_divergence(q, D) - jnp.sum(q * safe_log(A[observation]))
    assert np.allclose(float(free_energy), float(expected), atol=1e-6)

