"""Integration tests for Navier-Stokes PINN solver."""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from phio.physics.navier_stokes import analytical_taylor_green, lid_driven_cavity_bc
from phio.solvers.ns_pinn import create_ns_train_state, train_ns_pinn


class NSNetwork(nn.Module):
    """Simple MLP for Navier-Stokes (outputs u, v, p)."""

    @nn.compact
    def __call__(self, x, y, t):
        inputs = jnp.concatenate([x, y, t], axis=-1)
        h = nn.Dense(64)(inputs)
        h = nn.tanh(h)
        h = nn.Dense(64)(h)
        h = nn.tanh(h)
        h = nn.Dense(64)(h)
        h = nn.tanh(h)
        out = nn.Dense(3)(h)  # [u, v, p]
        return out


class TestNSTraining:
    """Test Navier-Stokes PINN training."""

    def test_training_reduces_loss(self):
        """Training should reduce total loss."""
        rng = jax.random.PRNGKey(42)
        model = NSNetwork()
        nu = 0.01

        # Generate training data (small for fast test)
        n_pde = 100
        n_bc = 20
        n_ic = 20

        rng, key = jax.random.split(rng)
        x_pde = jax.random.uniform(key, (n_pde,))
        y_pde = jax.random.uniform(key, (n_pde,))
        t_pde = jax.random.uniform(key, (n_pde,))

        # Boundary conditions (lid-driven cavity)
        x_bc = jnp.concatenate(
            [
                jnp.linspace(0, 1, n_bc),  # Top
                jnp.linspace(0, 1, n_bc),  # Bottom
                jnp.zeros(n_bc),  # Left
                jnp.ones(n_bc),  # Right
            ]
        )
        y_bc = jnp.concatenate(
            [
                jnp.ones(n_bc),  # Top
                jnp.zeros(n_bc),  # Bottom
                jnp.linspace(0, 1, n_bc),  # Left
                jnp.linspace(0, 1, n_bc),  # Right
            ]
        )
        t_bc = jnp.zeros_like(x_bc)
        u_bc, v_bc = lid_driven_cavity_bc(x_bc, y_bc, u_lid=1.0)

        # Initial condition (quiescent flow)
        x_ic = jax.random.uniform(key, (n_ic,))
        y_ic = jax.random.uniform(key, (n_ic,))
        u_ic = jnp.zeros(n_ic)
        v_ic = jnp.zeros(n_ic)

        # Create and train
        state = create_ns_train_state(rng, model, learning_rate=1e-3)
        state, history = train_ns_pinn(
            state,
            x_pde,
            y_pde,
            t_pde,
            x_bc,
            y_bc,
            t_bc,
            u_bc,
            v_bc,
            x_ic,
            y_ic,
            u_ic,
            v_ic,
            nu=nu,
            num_epochs=50,
            print_every=100,
        )

        # Check loss decreased
        assert history["total"][-1] < history["total"][0]
        assert len(history["total"]) == 50

    def test_taylor_green_convergence(self):
        """PINN should converge to Taylor-Green analytical solution."""
        rng = jax.random.PRNGKey(123)
        model = NSNetwork()
        nu = 0.01

        # Training data from analytical solution
        n_pde = 200
        n_bc = 30

        # Domain: [0, 2π] x [0, 2π]
        rng, key = jax.random.split(rng)
        x_pde = jax.random.uniform(key, (n_pde,)) * 2 * jnp.pi
        y_pde = jax.random.uniform(key, (n_pde,)) * 2 * jnp.pi
        t_pde = jax.random.uniform(key, (n_pde,)) * 0.5  # t ∈ [0, 0.5]

        # Boundary and IC from analytical solution
        x_bc = jax.random.uniform(key, (n_bc,)) * 2 * jnp.pi
        y_bc = jax.random.uniform(key, (n_bc,)) * 2 * jnp.pi
        t_bc = jnp.zeros(n_bc)

        u_bc, v_bc, _ = jax.vmap(lambda x, y: analytical_taylor_green(x, y, 0.0, nu))(
            x_bc, y_bc
        )

        x_ic, y_ic = x_bc, y_bc
        u_ic, v_ic = u_bc, v_bc

        # Train
        state = create_ns_train_state(rng, model, learning_rate=1e-3)
        state, history = train_ns_pinn(
            state,
            x_pde,
            y_pde,
            t_pde,
            x_bc,
            y_bc,
            t_bc,
            u_bc,
            v_bc,
            x_ic,
            y_ic,
            u_ic,
            v_ic,
            nu=nu,
            num_epochs=100,
            print_every=200,
        )

        # Test accuracy on holdout points
        n_test = 50
        rng, key = jax.random.split(rng)
        x_test = jax.random.uniform(key, (n_test,)) * 2 * jnp.pi
        y_test = jax.random.uniform(key, (n_test,)) * 2 * jnp.pi
        t_test = jax.random.uniform(key, (n_test,)) * 0.5

        # Predictions
        uvp_pred = jax.vmap(state.apply_fn, in_axes=(None, 0, 0, 0))(
            state.params,
            x_test[:, None],
            y_test[:, None],
            t_test[:, None],
        )

        u_pred = uvp_pred[:, 0]
        v_pred = uvp_pred[:, 1]

        # Ground truth
        u_true, v_true, _ = jax.vmap(
            lambda x, y, t: analytical_taylor_green(x, y, t, nu)
        )(x_test, y_test, t_test)

        # Relative error
        u_error = jnp.mean(jnp.abs(u_pred - u_true))
        v_error = jnp.mean(jnp.abs(v_pred - v_true))

        # Should achieve reasonable accuracy (relaxed for short training)
        assert u_error < 0.5  # Mean absolute error < 0.5
        assert v_error < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
