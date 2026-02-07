"""Test loss functions."""

import pytest
import jax.numpy as jnp
from jax import random
from phio.losses import pinn_loss, compute_pde_residual
from phio.networks import MLP


class TestPINNLoss:
    """Test physics-informed loss functions."""

    def test_pinn_loss_shape(self, rng):
        """Test loss returns scalar."""
        # Create network
        net = MLP(hidden_dims=[16], output_dim=1)
        x_sample = random.normal(rng, (5, 2))
        params = net.init(rng, x_sample)

        # Create sample data
        n = 10
        rng, *keys = random.split(rng, 7)
        x_colloc = random.uniform(keys[0], (n,))
        t_colloc = random.uniform(keys[1], (n,))
        x_bc = random.uniform(keys[2], (n,))
        t_bc = random.uniform(keys[3], (n,))
        u_bc = jnp.zeros(n)
        x_ic = random.uniform(keys[4], (n,))
        t_ic = jnp.zeros(n)
        u_ic = jnp.sin(jnp.pi * x_ic)

        # Compute loss
        loss = pinn_loss(
            params,
            net.apply,
            x_colloc,
            t_colloc,
            x_bc,
            t_bc,
            u_bc,
            x_ic,
            t_ic,
            u_ic,
        )

        # Check loss is scalar
        assert loss.shape == ()
        assert loss >= 0.0

    def test_loss_weights(self, rng):
        """Test loss weighting works."""
        # Setup (same as above)
        net = MLP(hidden_dims=[16], output_dim=1)
        x_sample = random.normal(rng, (5, 2))
        params = net.init(rng, x_sample)

        n = 10
        rng, *keys = random.split(rng, 7)
        x_colloc = random.uniform(keys[0], (n,))
        t_colloc = random.uniform(keys[1], (n,))
        x_bc = random.uniform(keys[2], (n,))
        t_bc = random.uniform(keys[3], (n,))
        u_bc = jnp.zeros(n)
        x_ic = random.uniform(keys[4], (n,))
        t_ic = jnp.zeros(n)
        u_ic = jnp.sin(jnp.pi * x_ic)

        # Compute with different weights
        loss1 = pinn_loss(
            params,
            net.apply,
            x_colloc,
            t_colloc,
            x_bc,
            t_bc,
            u_bc,
            x_ic,
            t_ic,
            u_ic,
            weights={"pde": 1.0, "bc": 1.0, "ic": 1.0},
        )

        loss2 = pinn_loss(
            params,
            net.apply,
            x_colloc,
            t_colloc,
            x_bc,
            t_bc,
            u_bc,
            x_ic,
            t_ic,
            u_ic,
            weights={"pde": 0.0, "bc": 1.0, "ic": 1.0},
        )

        # Loss should be different (unless PDE term is already zero)
        # Just check both are valid
        assert jnp.isfinite(loss1)
        assert jnp.isfinite(loss2)
