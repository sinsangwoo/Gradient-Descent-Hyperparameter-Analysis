"""Test utility functions."""

import pytest
import jax.numpy as jnp
from phio.utils import compute_l2_error, compute_metrics


class TestMetrics:
    """Test metric computations."""

    def test_l2_error_zero(self):
        """Test L2 error is zero for identical arrays."""
        u = jnp.ones((10, 10))
        error = compute_l2_error(u, u, relative=True)
        assert jnp.isclose(error, 0.0)

    def test_l2_error_nonzero(self):
        """Test L2 error is positive for different arrays."""
        u_pred = jnp.ones((10, 10))
        u_exact = jnp.ones((10, 10)) * 2.0
        error = compute_l2_error(u_pred, u_exact, relative=True)
        assert error > 0.0
        assert error < 1.0  # Should be 0.5 for this case

    def test_compute_metrics_keys(self):
        """Test metrics dict has expected keys."""
        u_pred = jnp.ones((10, 10))
        u_exact = jnp.ones((10, 10)) * 1.1
        metrics = compute_metrics(u_pred, u_exact)

        expected_keys = {"l2_relative", "l2_absolute", "max_error", "mean_error"}
        assert set(metrics.keys()) == expected_keys

        # All metrics should be positive
        for value in metrics.values():
            assert value >= 0.0
