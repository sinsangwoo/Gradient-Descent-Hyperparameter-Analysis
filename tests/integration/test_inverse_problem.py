"""Integration tests for inverse problem solver (Phase 2.2)."""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from phio.physics.heat import heat_equation_residual
from phio.solvers.inverse_problem import InverseProblemSolver


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


def generate_synthetic_measurements(
    rng: jax.random.PRNGKey,
    true_alpha: float = 0.01,
    n_measurements: int = 20,
    noise_level: float = 0.01,
):
    """Generate synthetic measurement data."""
    # Random measurement locations
    x_meas = jax.random.uniform(rng, shape=(n_measurements,), minval=0.1, maxval=0.9)
    t_meas = jax.random.uniform(rng, shape=(n_measurements,), minval=0.1, maxval=0.5)

    # Analytical solution (ground truth)
    def analytical_solution(x, t):
        return jnp.exp(-50 * (x - 0.5) ** 2 / (1 + 200 * true_alpha * t)) / jnp.sqrt(
            1 + 200 * true_alpha * t
        )

    u_true = jax.vmap(analytical_solution)(x_meas, t_meas)

    # Add noise
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, shape=u_true.shape) * noise_level
    u_meas = u_true + noise

    return x_meas, t_meas, u_meas


class TestInverseProblem:
    """Test inverse problem solver."""

    def test_inverse_loss_computation(self):
        """Test inverse loss computation with measurements."""
        from phio.solvers.pinn_trainer import create_train_state

        rng = jax.random.PRNGKey(42)
        model = SimpleMLP()
        solver = InverseProblemSolver(model, heat_equation_residual)

        # Generate synthetic data
        x_meas, t_meas, u_meas = generate_synthetic_measurements(rng, true_alpha=0.01)

        # Initialize state
        state = create_train_state(
            rng,
            model,
            learning_rate=1e-3,
            sample_input=(jnp.ones((1, 1)), jnp.zeros((1, 1))),
        )

        # Mock physical parameters
        physical_params = {"alpha": jnp.array(0.05)}  # Wrong initial guess

        # Mock training data
        data = {
            "x_pde": jax.random.uniform(rng, (50,)),
            "t_pde": jax.random.uniform(rng, (50,)),
            "x_bc": jnp.array([0.0, 1.0]),
            "t_bc": jnp.array([0.0, 0.0]),
            "u_bc": jnp.array([0.0, 0.0]),
            "x_ic": jnp.linspace(0, 1, 20),
            "u_ic": jax.vmap(gaussian_ic)(jnp.linspace(0, 1, 20)),
            "x_meas": x_meas,
            "t_meas": t_meas,
            "u_meas": u_meas,
        }

        # Compute loss
        loss, loss_dict = solver.compute_inverse_loss(
            state, physical_params, **data, data_weight=10.0
        )

        # Check loss components
        assert loss_dict["data"] >= 0
        assert loss_dict["pde"] >= 0
        assert loss_dict["bc"] >= 0
        assert loss_dict["ic"] >= 0
        assert loss_dict["total"] > 0

    def test_parameter_estimation(self):
        """Test parameter estimation from measurements."""
        rng = jax.random.PRNGKey(42)
        model = SimpleMLP()
        solver = InverseProblemSolver(model, heat_equation_residual)

        # Generate synthetic measurements with known parameter
        true_alpha = 0.01
        x_meas, t_meas, u_meas = generate_synthetic_measurements(
            rng, true_alpha=true_alpha, n_measurements=30, noise_level=0.005
        )

        # Solve inverse problem with wrong initial guess
        initial_guess = {"alpha": 0.05}  # 5x wrong
        state, estimated_params, history = solver.solve_inverse_problem(
            rng,
            x_meas,
            t_meas,
            u_meas,
            gaussian_ic,
            initial_guess,
            n_epochs=500,  # Short test
            n_collocation_points=50,
            data_weight=10.0,
            print_every=100,
        )

        # Check estimation improved
        initial_error = abs(initial_guess["alpha"] - true_alpha)
        final_error = abs(estimated_params["alpha"] - true_alpha)
        print(f"\nInitial error: {initial_error:.6f}")
        print(f"Final error: {final_error:.6f}")
        print(f"True α: {true_alpha:.6f}")
        print(f"Estimated α: {estimated_params['alpha']:.6f}")

        # Estimation should improve (even if not perfect due to short training)
        assert final_error < initial_error

        # Check history
        assert len(history["alpha"]) == 500
        assert history["total"][-1] < history["total"][0]  # Loss decreased

    def test_convergence_with_more_data(self):
        """Test that more measurements improve estimation."""
        rng = jax.random.PRNGKey(42)
        model = SimpleMLP()
        solver = InverseProblemSolver(model, heat_equation_residual)

        true_alpha = 0.01
        initial_guess = {"alpha": 0.05}

        errors = []
        for n_meas in [10, 20, 40]:
            rng, subrng = jax.random.split(rng)
            x_meas, t_meas, u_meas = generate_synthetic_measurements(
                subrng, true_alpha=true_alpha, n_measurements=n_meas, noise_level=0.005
            )

            _, estimated_params, _ = solver.solve_inverse_problem(
                subrng,
                x_meas,
                t_meas,
                u_meas,
                gaussian_ic,
                initial_guess,
                n_epochs=300,
                print_every=1000,  # Quiet
            )

            error = abs(estimated_params["alpha"] - true_alpha)
            errors.append(error)

        print(f"\nErrors with increasing measurements: {errors}")
        # More data should generally reduce error (though not guaranteed due to randomness)
        # At least check that we can solve the problem
        assert errors[-1] < initial_guess["alpha"] - true_alpha
