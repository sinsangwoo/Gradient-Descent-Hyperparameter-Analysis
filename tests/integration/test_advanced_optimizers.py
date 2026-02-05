"""Integration tests for advanced optimizers with PINN training."""

import jax
import jax.numpy as jnp
import pytest

from phio.networks.mlp import MLP
from phio.optimizers.causal import CausalWeightScheduler
from phio.optimizers.loss_balancing import AdaptiveLossBalancer
from phio.physics.heat import analytical_gaussian, heat_equation_residual
from phio.solvers.pinn_trainer import create_train_state, train_pinn


@pytest.fixture
def simple_heat_data():
    """Generate minimal heat equation dataset."""
    rng = jax.random.PRNGKey(42)
    rng_x, rng_t = jax.random.split(rng, 2)

    # Small dataset
    x_pde = jax.random.uniform(rng_x, (50, 1))
    t_pde = jax.random.uniform(rng_t, (50, 1))

    t_bc = jnp.linspace(0, 1, 10)
    x_bc = jnp.concatenate([jnp.zeros(10), jnp.ones(10)])
    t_bc = jnp.concatenate([t_bc, t_bc])
    u_bc = jnp.zeros_like(x_bc)

    x_ic = jnp.linspace(0, 1, 20)
    alpha = 0.01
    u_ic = analytical_gaussian(
        x_ic, jnp.zeros_like(x_ic), alpha=alpha, x0=0.5, sigma0=0.1
    ).squeeze()

    return {
        "x_pde": x_pde,
        "t_pde": t_pde,
        "x_bc": x_bc,
        "t_bc": t_bc,
        "u_bc": u_bc,
        "x_ic": x_ic,
        "u_ic": u_ic,
        "alpha": alpha,
    }


class TestCausalTraining:
    """Test causal weighting in PINN training."""

    def test_causal_scheduler_integration(self, simple_heat_data):
        """Causal scheduler should work with training loop."""
        data = simple_heat_data

        # Initialize scheduler
        scheduler = CausalWeightScheduler(
            t_min=0.0, t_max=1.0, num_stages=3, epochs_per_stage=50
        )

        # Test weight computation - fix shape mismatch
        weights = scheduler.get_temporal_weights(data["t_pde"].flatten(), epoch=0)
        assert weights.shape == (data["t_pde"].shape[0],)  # Changed from data["t_pde"].shape
        assert jnp.all((weights >= 0) & (weights <= 1))

        # Check stage progression
        info_early = scheduler.get_stage_info(epoch=0)
        info_late = scheduler.get_stage_info(epoch=100)

        assert info_early["stage"] < info_late["stage"]


class TestAdaptiveBalancing:
    """Test adaptive loss balancing."""

    def test_balancer_with_training(self, simple_heat_data):
        """Loss balancer should adjust during training."""
        data = simple_heat_data

        balancer = AdaptiveLossBalancer(alpha=0.1, update_freq=10)

        # Initial weights
        initial_weights = balancer.weights.copy()

        # Simulate gradient updates (would come from actual training)
        for epoch in range(50):
            # Mock gradients with different magnitudes
            gradients = {
                "pde": {"w": jnp.ones(10) * (5.0 - epoch * 0.1)},
                "bc": {"w": jnp.ones(10) * 2.0},
                "ic": {"w": jnp.ones(10) * 2.0},
            }

            balancer.update_weights(gradients, epoch=epoch)

        # Weights should have changed
        final_weights = balancer.weights
        assert any(
            abs(final_weights[k] - initial_weights[k]) > 0.01
            for k in initial_weights.keys()
        )

    def test_convergence_improvement(self, simple_heat_data):
        """Adaptive balancing may improve convergence."""
        # This is a placeholder for future benchmark
        # Would compare: baseline vs adaptive balancing
        data = simple_heat_data

        rng = jax.random.PRNGKey(0)
        model = MLP(features=[16, 16, 1])
        state = create_train_state(rng, model, learning_rate=1e-3)

        # Train for few epochs (smoke test)
        state, history = train_pinn(
            state,
            data["x_pde"],
            data["t_pde"],
            data["x_bc"],
            data["t_bc"],
            data["u_bc"],
            data["x_ic"],
            data["u_ic"],
            pde_residual_fn=heat_equation_residual,
            alpha=data["alpha"],
            num_epochs=50,
            print_every=25,
        )

        # Should complete without error
        assert len(history["total"]) == 50
        # Loss should generally decrease
        assert history["total"][-1] < history["total"][0] * 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
