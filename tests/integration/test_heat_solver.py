"""Integration tests for complete PINN heat equation solver."""

import jax
import jax.numpy as jnp
import pytest

from phio.networks.mlp import FourierFeatureMLP, MLP
from phio.physics.heat import analytical_gaussian, heat_equation_residual
from phio.solvers.pinn_trainer import create_train_state, train_pinn


@pytest.fixture
def sample_training_data():
    """Generate small training dataset for fast testing."""
    rng = jax.random.PRNGKey(42)

    # Small dataset for fast tests
    n_pde, n_bc, n_ic = 100, 20, 20
    alpha = 0.01

    rng, rng_x, rng_t = jax.random.split(rng, 3)
    x_pde = jax.random.uniform(rng_x, (n_pde, 1))
    t_pde = jax.random.uniform(rng_t, (n_pde, 1))

    # Boundary conditions
    t_bc = jnp.linspace(0, 1, n_bc)
    x_bc = jnp.concatenate([jnp.zeros(n_bc // 2), jnp.ones(n_bc // 2)])
    t_bc = jnp.concatenate([t_bc[: n_bc // 2], t_bc[: n_bc // 2]])
    u_bc = jnp.zeros_like(x_bc)

    # Initial condition
    x_ic = jnp.linspace(0, 1, n_ic)
    u_ic = analytical_gaussian(x_ic, jnp.zeros_like(x_ic), alpha=alpha, x0=0.5, sigma0=0.1)
    u_ic = u_ic.squeeze()

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


class TestPINNTraining:
    """Test PINN training pipeline."""

    def test_training_reduces_loss(self, sample_training_data):
        """Training should reduce total loss."""
        data = sample_training_data

        # Initialize
        rng = jax.random.PRNGKey(0)
        model = MLP(features=[32, 32, 1])
        state = create_train_state(rng, model, learning_rate=1e-3)

        # Train for few epochs
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
            num_epochs=100,
            print_every=50,
        )

        # Loss should decrease
        initial_loss = history["total"][0]
        final_loss = history["total"][-1]
        assert final_loss < initial_loss

    def test_curriculum_learning(self, sample_training_data):
        """Curriculum learning should apply weight schedule."""
        data = sample_training_data

        rng = jax.random.PRNGKey(1)
        model = MLP(features=[32, 32, 1])
        state = create_train_state(rng, model)

        # Curriculum: prioritize IC early
        curriculum = {
            0: {"ic": 10.0, "bc": 1.0, "pde": 0.1},
            50: {"ic": 1.0, "bc": 1.0, "pde": 1.0},
        }

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
            num_epochs=100,
            curriculum_schedule=curriculum,
        )

        # IC loss should decrease faster early on
        ic_loss_reduction_early = history["ic"][0] - history["ic"][49]
        ic_loss_reduction_late = history["ic"][50] - history["ic"][-1]

        # With high IC weight early, IC loss should drop more in first half
        assert ic_loss_reduction_early > 0  # IC loss decreases

    @pytest.mark.parametrize("model_class", [MLP, FourierFeatureMLP])
    def test_different_architectures(self, model_class, sample_training_data):
        """Different network architectures should train successfully."""
        data = sample_training_data

        rng = jax.random.PRNGKey(2)
        if model_class == FourierFeatureMLP:
            model = model_class(features=[32, 32, 1], fourier_features=16)
        else:
            model = model_class(features=[32, 32, 1])

        state = create_train_state(rng, model)

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
        )

        # Should converge
        assert history["total"][-1] < history["total"][0]


class TestAccuracy:
    """Test prediction accuracy against analytical solution."""

    @pytest.mark.slow
    def test_convergence_to_analytical(self, sample_training_data):
        """PINN should converge to analytical solution with sufficient training."""
        data = sample_training_data

        # Train with more epochs for better accuracy
        rng = jax.random.PRNGKey(3)
        model = FourierFeatureMLP(features=[64, 64, 64, 1], fourier_features=32)
        state = create_train_state(rng, model)

        state, _ = train_pinn(
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
            num_epochs=2000,
        )

        # Test on grid
        x_test = jnp.linspace(0, 1, 50)
        t_test = jnp.linspace(0, 1, 20)
        X, T = jnp.meshgrid(x_test, t_test)

        # Predictions
        u_pred = jax.vmap(
            jax.vmap(state.apply_fn, in_axes=(None, 0, None)), in_axes=(None, None, 0)
        )(state.params, X, T).squeeze()

        # Analytical
        u_true = analytical_gaussian(x_test, t_test, alpha=data["alpha"])

        # L2 error
        l2_error = jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true)

        # Should achieve < 1% relative error with enough training
        assert l2_error < 0.01, f"L2 error {l2_error:.6f} exceeds threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
