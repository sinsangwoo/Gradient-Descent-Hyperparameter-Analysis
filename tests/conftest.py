"""Pytest configuration and fixtures."""

import pytest
import jax
import jax.numpy as jnp
from jax import random


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(seed):
    """JAX random number generator."""
    return random.PRNGKey(seed)


@pytest.fixture
def sample_1d_grid():
    """Sample 1D spatial-temporal grid."""
    x = jnp.linspace(0, 1, 50)
    t = jnp.linspace(0, 1, 50)
    return x, t


@pytest.fixture
def sample_heat_solution(sample_1d_grid):
    """Exact solution to 1D heat equation."""
    x, t = sample_1d_grid
    X, T = jnp.meshgrid(x, t)
    alpha = 0.01
    u_exact = jnp.exp(-alpha * jnp.pi**2 * T) * jnp.sin(jnp.pi * X)
    return u_exact
