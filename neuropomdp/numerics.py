"""Numerically stable helpers for categorical inference and planning."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .types import Array


def configure_jax_precision(enable_x64: bool) -> None:
    """Configure JAX precision before arrays are materialized."""

    jax.config.update("jax_enable_x64", enable_x64)


@jax.jit
def safe_log(x: Array, eps: float = 1e-8) -> Array:
    """Compute a clipped logarithm that never returns ``-inf``."""

    return jnp.log(jnp.clip(x, eps, None))


@partial(jax.jit, static_argnames=("axis",))
def normalize(probabilities: Array, axis: int = -1, eps: float = 1e-8) -> Array:
    """Normalize an array along ``axis`` while avoiding division by zero."""

    total = jnp.sum(probabilities, axis=axis, keepdims=True)
    normalized = probabilities / jnp.clip(total, eps, None)
    size = probabilities.shape[axis]
    uniform = jnp.full_like(probabilities, 1.0 / float(size))
    return jnp.where(total > eps, normalized, uniform)


@partial(jax.jit, static_argnames=("axis",))
def entropy(probabilities: Array, axis: int = -1) -> Array:
    """Compute Shannon entropy for a categorical distribution."""

    probs = jnp.clip(probabilities, 1e-8, None)
    return -jnp.sum(probs * jnp.log(probs), axis=axis)


@partial(jax.jit, static_argnames=("axis",))
def kl_divergence(q: Array, p: Array, axis: int = -1) -> Array:
    """Compute ``KL(q || p)`` for categorical distributions."""

    q_safe = jnp.clip(q, 1e-8, None)
    p_safe = jnp.clip(p, 1e-8, None)
    return jnp.sum(q_safe * (jnp.log(q_safe) - jnp.log(p_safe)), axis=axis)


@jax.jit
def softmax(logits: Array, axis: int = -1) -> Array:
    """Stable softmax wrapper."""

    return jax.nn.softmax(logits, axis=axis)


@jax.jit
def categorical_sample(key: Array, probabilities: Array) -> Array:
    """Sample an index from a categorical distribution."""

    logits = jnp.log(jnp.clip(probabilities, 1e-8, None))
    return jax.random.categorical(key, logits)


@jax.jit
def one_hot(index: Array, size: int, dtype: jnp.dtype = jnp.float32) -> Array:
    """Return a one-hot vector."""

    return jax.nn.one_hot(index, size, dtype=dtype)
