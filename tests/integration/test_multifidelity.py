"""Integration tests for multi-fidelity optimization (Phase 2.2)."""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from phio.solvers.multifidelity import MultiFidelitySolver


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    @nn.compact
    def __call__(self, x, t):
        inputs = jnp.concatenate([x, t], axis=-1)
        x = nn.Dense(32)(inputs)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


def gaussian_ic(x: jnp.ndarray) -> jnp.ndarray:
    """Gaussian initial condition."""
    return jnp.exp(-50 * (x - 0.5) ** 2)


def analytical_heat_solution(
    x: jnp.ndarray, t: jnp.ndarray, alpha: float = 0.01
) -> jnp.ndarray:
    """Analytical solution for heat equation with Gaussian IC."""
    return jnp.exp(-50 * (x - 0.5) ** 2 / (1 + 200 * alpha * t)) / jnp.sqrt(
        1 + 200 * alpha * t
    )


class TestMultiFidelity:
    """Test multi-fidelity optimization pipeline."""

    def test_low_fidelity_training(self):
        """Test low-fidelity training on coarse grid."""
        rng = jax.random.PRNGKey(42)
        model = SimpleMLP()
        solver = MultiFidelitySolver(model, alpha=0.01)

        state, history, elapsed_time = solver.train_low_fidelity(rng, gaussian_ic)

        # Check training completed
        assert state is not None
        assert elapsed_time > 0
        assert len(history["total"]) == solver.low_fidelity.n_epochs

        # Check loss decreased
        assert history["total"][-1] < history["total"][0]

    def test_high_fidelity_refinement(self):
        """Test high-fidelity refinement after low-fidelity."""
        rng = jax.random.PRNGKey(42)
        model = SimpleMLP()
        solver = MultiFidelitySolver(model, alpha=0.01)

        # Train low-fidelity first
        low_state, _, _ = solver.train_low_fidelity(rng, gaussian_ic)

        # Refine with high-fidelity
        high_state, history, elapsed_time = solver.refine_with_high_fidelity(
            low_state, gaussian_ic
        )

        assert high_state is not None
        assert elapsed_time > 0
        assert len(history["total"]) == solver.high_fidelity.n_epochs

    def test_accuracy_computation(self):
        """Test accuracy metrics computation."""
        rng = jax.random.PRNGKey(42)
        model = SimpleMLP()
        solver = MultiFidelitySolver(model, alpha=0.01)

        state, _, _ = solver.train_low_fidelity(rng, gaussian_ic)

        # Compute accuracy
        accuracy = solver.compute_accuracy(
            state,
            lambda x, t: analytical_heat_solution(x, t, alpha=0.01),
            n_test_points=50,
        )

        # Check metrics exist and are reasonable
        assert "mse" in accuracy
        assert "mae" in accuracy
        assert "relative_error" in accuracy
        assert 0 <= accuracy["relative_error"] <= 1

    def test_complete_pipeline(self):
        """Test complete multi-fidelity pipeline."""
        rng = jax.random.PRNGKey(42)
        model = SimpleMLP()
        solver = MultiFidelitySolver(model, alpha=0.01)

        # Run complete pipeline
        results = solver.multifidelity_pipeline(
            rng,
            gaussian_ic,
            lambda x, t: analytical_heat_solution(x, t, alpha=0.01),
        )

        # Check all stages completed
        assert "low_fidelity" in results
        assert "high_fidelity" in results
        assert "cost_function" in results
        assert "error_reduction_percent" in results

        # Check improvement
        low_error = results["low_fidelity"]["accuracy"]["relative_error"]
        high_error = results["high_fidelity"]["accuracy"]["relative_error"]
        assert high_error < low_error  # High-fidelity should be more accurate

        # Check cost function is computed
        assert results["cost_function"] > 0
