"""Exact Bayesian state inference and variational free energy."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .numerics import entropy, kl_divergence, normalize, safe_log
from .types import InferenceResult


@jax.jit
def posterior_from_observation(prior: jnp.ndarray, likelihood_column: jnp.ndarray) -> jnp.ndarray:
    """Compute the exact posterior ``q(s|o)`` up to normalization."""

    return normalize(prior * likelihood_column)


def exact_posterior(A: jnp.ndarray, D: jnp.ndarray, observation: int) -> jnp.ndarray:
    """Compute the exact posterior for a single categorical observation."""

    return posterior_from_observation(D, A[observation])


def variational_free_energy(
    q: jnp.ndarray,
    prior: jnp.ndarray,
    likelihood_column: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the exact free energy decomposition.

    ``F = KL(q || prior) - E_q[log P(o|s)]``
    """

    expected_log_likelihood = jnp.sum(q * safe_log(likelihood_column))
    kl = kl_divergence(q, prior)
    return kl - expected_log_likelihood


def inference_diagnostics(
    prior: jnp.ndarray,
    posterior: jnp.ndarray,
    likelihood_column: jnp.ndarray,
) -> InferenceResult:
    """Return Bayesian inference diagnostics for one update."""

    expected_log_likelihood = jnp.sum(posterior * safe_log(likelihood_column))
    kl = kl_divergence(posterior, prior)
    free_energy = kl - expected_log_likelihood
    return InferenceResult(
        prior=prior,
        posterior=posterior,
        free_energy=free_energy,
        kl_divergence=kl,
        expected_log_likelihood=expected_log_likelihood,
        prior_entropy=entropy(prior),
        posterior_entropy=entropy(posterior),
    )

