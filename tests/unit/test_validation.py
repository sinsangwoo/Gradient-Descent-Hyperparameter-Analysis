"""Unit tests for validation tools."""

import jax.numpy as jnp
import pytest

from phio.datasets.ghia_cavity import GhiaCavityData
from phio.validation.metrics import compute_error_metrics, generate_error_report


class TestGhiaBenchmark:
    """Test Ghia benchmark dataset."""

    def test_re100_data_shape(self):
        """Re=100 data should have correct shape."""
        data = GhiaCavityData.get_data(100)

        assert len(data["y_coords"]) == 17
        assert len(data["u_velocity"]) == 17
        assert len(data["x_coords"]) == 17
        assert len(data["v_velocity"]) == 17

    def test_re400_data_available(self):
        """Re=400 data should be available."""
        data = GhiaCavityData.get_data(400)

        assert "y_coords" in data
        assert "u_velocity" in data
        assert "x_coords" in data
        assert "v_velocity" in data

    def test_re1000_data_available(self):
        """Re=1000 data should be available."""
        data = GhiaCavityData.get_data(1000)

        assert "y_coords" in data
        assert "u_velocity" in data

    def test_invalid_reynolds_number(self):
        """Invalid Reynolds number should raise error."""
        with pytest.raises(ValueError):
            GhiaCavityData.get_data(500)

    def test_boundary_conditions(self):
        """Boundary values should be physically reasonable."""
        data = GhiaCavityData.get_data(100)

        # Top wall (y=1) should have u close to 1.0
        top_idx = jnp.argmax(data["y_coords"])
        assert jnp.abs(data["u_velocity"][top_idx] - 1.0) < 0.01

        # Bottom wall (y=0) should have u close to 0.0
        bottom_idx = jnp.argmin(data["y_coords"])
        assert jnp.abs(data["u_velocity"][bottom_idx]) < 0.01


class TestErrorMetrics:
    """Test error metric computation."""

    def test_perfect_prediction(self):
        """Zero error for perfect prediction."""
        pred = jnp.array([1.0, 2.0, 3.0])
        truth = jnp.array([1.0, 2.0, 3.0])

        metrics = compute_error_metrics(pred, truth)

        assert metrics["mae"] < 1e-10
        assert metrics["mse"] < 1e-10
        assert metrics["rmse"] < 1e-10
        assert metrics["max_error"] < 1e-10
        assert metrics["relative_l2"] < 1e-10

    def test_known_error(self):
        """Known error values should be computed correctly."""
        pred = jnp.array([1.0, 2.0, 3.0])
        truth = jnp.array([0.0, 0.0, 0.0])

        metrics = compute_error_metrics(pred, truth)

        assert jnp.abs(metrics["mae"] - 2.0) < 1e-6
        assert jnp.abs(metrics["max_error"] - 3.0) < 1e-6

    def test_relative_l2_scale_invariance(self):
        """Relative L2 should be scale-invariant."""
        pred1 = jnp.array([1.0, 2.0, 3.0])
        truth1 = jnp.array([1.1, 2.1, 3.1])

        pred2 = pred1 * 10
        truth2 = truth1 * 10

        metrics1 = compute_error_metrics(pred1, truth1)
        metrics2 = compute_error_metrics(pred2, truth2)

        # Relative L2 should be approximately equal
        assert jnp.abs(metrics1["relative_l2"] - metrics2["relative_l2"]) < 0.01


class TestErrorReport:
    """Test error report generation."""

    def test_report_format(self):
        """Report should contain key sections."""
        u_metrics = {
            "mae": 0.01,
            "rmse": 0.015,
            "max_error": 0.05,
            "relative_l2": 0.02,
        }
        v_metrics = {
            "mae": 0.012,
            "rmse": 0.018,
            "max_error": 0.06,
            "relative_l2": 0.025,
        }

        report = generate_error_report(u_metrics, v_metrics, 100)

        assert "Re = 100" in report
        assert "U-Velocity" in report
        assert "V-Velocity" in report
        assert "Overall Assessment" in report
        assert "EXCELLENT" in report  # <1% error

    def test_quality_classification(self):
        """Quality should be classified correctly."""
        # Excellent case (<1%)
        u1 = {"mae": 0.005, "rmse": 0.008, "max_error": 0.02, "relative_l2": 0.008}
        v1 = {"mae": 0.005, "rmse": 0.008, "max_error": 0.02, "relative_l2": 0.008}
        report1 = generate_error_report(u1, v1, 100)
        assert "EXCELLENT" in report1

        # Good case (1-5%)
        u2 = {"mae": 0.03, "rmse": 0.04, "max_error": 0.1, "relative_l2": 0.03}
        v2 = {"mae": 0.03, "rmse": 0.04, "max_error": 0.1, "relative_l2": 0.03}
        report2 = generate_error_report(u2, v2, 100)
        assert "GOOD" in report2

        # Acceptable case (5-10%)
        u3 = {"mae": 0.07, "rmse": 0.09, "max_error": 0.2, "relative_l2": 0.07}
        v3 = {"mae": 0.07, "rmse": 0.09, "max_error": 0.2, "relative_l2": 0.07}
        report3 = generate_error_report(u3, v3, 100)
        assert "ACCEPTABLE" in report3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
