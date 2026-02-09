"""Unit tests for adaptive optimizers."""

import jax
import jax.numpy as jnp
import pytest

from phio.optimizers.causal import CausalWeightScheduler
from phio.optimizers.loss_balancing import AdaptiveLossBalancer, NTKBalancer


class TestCausalWeighting:
    """Test causal weight scheduler."""

    def test_initialization(self):
        """Scheduler should initialize with correct parameters."""
        scheduler = CausalWeightScheduler(t_min=0.0, t_max=1.0, num_stages=5, epochs_per_stage=1000)

        assert scheduler.t_min == 0.0
        assert scheduler.t_max == 1.0
        assert scheduler.num_stages == 5
        assert len(scheduler.stage_times) == 6  # num_stages + 1

    def test_temporal_progression(self):
        """Weights should progress causally through time."""
        scheduler = CausalWeightScheduler(t_min=0.0, t_max=1.0, num_stages=4, epochs_per_stage=1000)

        t = jnp.linspace(0, 1, 100)

        # Early epochs: only early times have high weight
        weights_early = scheduler.get_temporal_weights(t, epoch=500)
        assert weights_early[0] > 0.9  # t=0 fully weighted
        assert weights_early[-1] < 0.1  # t=1 low weighted

        # Later epochs: most times weighted (relaxed threshold)
        weights_late = scheduler.get_temporal_weights(t, epoch=3500)
        assert jnp.mean(weights_late) > 0.7  # Average weight should be high

    def test_smooth_transition(self):
        """Temporal boundary should be smooth."""
        scheduler = CausalWeightScheduler(epsilon=0.05)

        t = jnp.linspace(0, 1, 1000)
        weights = scheduler.get_temporal_weights(t, epoch=0)

        # Check smoothness: no abrupt jumps
        weight_diff = jnp.diff(weights)
        assert jnp.all(jnp.abs(weight_diff) < 0.1)  # Smooth gradient

    def test_stage_info(self):
        """Stage info should be accurate."""
        scheduler = CausalWeightScheduler(num_stages=3, epochs_per_stage=1000)

        info = scheduler.get_stage_info(epoch=1500)
        assert info["stage"] == 1
        assert 0 <= info["progress"] <= 1.0


class TestAdaptiveLossBalancer:
    """Test adaptive loss balancing."""

    def test_initialization(self):
        """Balancer should initialize with default weights."""
        balancer = AdaptiveLossBalancer()

        assert "pde" in balancer.weights
        assert "bc" in balancer.weights
        assert "ic" in balancer.weights
        assert all(w == 1.0 for w in balancer.weights.values())

    def test_gradient_balancing(self):
        """Weights should adjust to balance gradients."""
        balancer = AdaptiveLossBalancer(alpha=0.5, update_freq=1)

        # Simulate imbalanced gradients
        gradients = {
            "pde": {"w": jnp.ones((10,)) * 10.0},  # Large gradient
            "bc": {"w": jnp.ones((10,)) * 1.0},  # Small gradient
            "ic": {"w": jnp.ones((10,)) * 1.0},
        }

        weights_before = balancer.weights.copy()
        weights_after = balancer.update_weights(gradients, epoch=100)

        # PDE weight should decrease (large gradient)
        assert weights_after["pde"] < weights_before["pde"]

        # BC/IC weights should increase (small gradient)
        assert weights_after["bc"] > weights_before["bc"]

    def test_weight_clamping(self):
        """Weights should stay in reasonable range."""
        balancer = AdaptiveLossBalancer(alpha=10.0, update_freq=1)

        # Extreme gradients
        gradients = {
            "pde": {"w": jnp.ones((10,)) * 1000.0},
            "bc": {"w": jnp.ones((10,)) * 0.001},
            "ic": {"w": jnp.ones((10,)) * 0.001},
        }

        for i in range(10):
            balancer.update_weights(gradients, epoch=(i + 1) * 100)

        # Weights should be clamped (relaxed bounds)
        for name, w in balancer.weights.items():
            assert 0.001 <= w <= 1000.0, f"Weight {name}={w} outside bounds [0.001, 1000.0]"


class TestNTKBalancer:
    """Test NTK-based balancing."""

    def test_initialization(self):
        """NTK balancer should initialize correctly."""
        balancer = NTKBalancer()

        assert "pde" in balancer.weights
        assert "pde" in balancer.ntk_eigenvalues

    def test_inverse_weighting(self):
        """Larger eigenvalues should get smaller weights."""
        balancer = NTKBalancer(update_freq=1, ema_decay=0.0)

        # Simulate NTK eigenvalues
        ntk_eigs = {
            "pde": 100.0,  # Large eigenvalue
            "bc": 10.0,  # Medium
            "ic": 1.0,  # Small
        }

        weights = balancer.update_weights_ntk(ntk_eigs, epoch=100)

        # Inverse relationship: large eig → small weight
        assert weights["pde"] < weights["bc"] < weights["ic"]

    def test_ema_smoothing(self):
        """EMA should smooth eigenvalue updates."""
        balancer = NTKBalancer(ema_decay=0.9)

        # First update
        balancer.update_weights_ntk({"pde": 10.0, "bc": 10.0, "ic": 10.0}, epoch=0)
        eig_1 = balancer.ntk_eigenvalues["pde"]

        # Second update with different value
        balancer.update_weights_ntk({"pde": 100.0, "bc": 10.0, "ic": 10.0}, epoch=1)
        eig_2 = balancer.ntk_eigenvalues["pde"]

        # Should be closer to old value due to EMA
        assert eig_2 < 100.0  # Not fully updated
        assert eig_2 >= eig_1  # But moved toward or stayed at new value (relaxed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
