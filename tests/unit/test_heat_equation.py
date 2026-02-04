"""Unit tests for heat equation physics module."""

import jax
import jax.numpy as jnp
import pytest

from phio.physics.heat import (
    analytical_gaussian,
    heat_equation_residual,
    steady_state_1d,
)


class TestAnalyticalSolutions:
    """Test analytical solutions for validation."""

    def test_gaussian_shape(self):
        """Gaussian solution should have correct shape."""
        x = jnp.linspace(0, 1, 50)
        t = jnp.linspace(0, 1, 20)
        u = analytical_gaussian(x, t)

        assert u.shape == (len(t), len(x))

    def test_gaussian_initial_condition(self):
        """Gaussian at t=0 should match initial condition."""
        x = jnp.linspace(0, 1, 100)
        t = jnp.array([0.0])
        x0, sigma0 = 0.5, 0.1

        u = analytical_gaussian(x, t, x0=x0, sigma0=sigma0)
        expected = jnp.exp(-(x - x0) ** 2 / (2 * sigma0**2))

        assert jnp.allclose(u[0, :], expected, atol=1e-6)

    def test_gaussian_diffusion(self):
        """Gaussian should spread over time (variance increases)."""
        x = jnp.linspace(0, 1, 100)
        t0, t1 = 0.0, 1.0
        alpha = 0.01

        u0 = analytical_gaussian(x, jnp.array([t0]), alpha=alpha)
        u1 = analytical_gaussian(x, jnp.array([t1]), alpha=alpha)

        # Peak should decrease (mass conservation + spreading)
        assert jnp.max(u1) < jnp.max(u0)

        # Width should increase (measured by standard deviation)
        var0 = jnp.sum(x**2 * u0[0, :]) / jnp.sum(u0[0, :])
        var1 = jnp.sum(x**2 * u1[0, :]) / jnp.sum(u1[0, :])
        assert var1 > var0

    def test_steady_state_linear(self):
        """Steady state should be linear between boundaries."""
        x = jnp.linspace(0, 1, 100)
        u_left, u_right = 0.0, 1.0

        u = steady_state_1d(x, u_left, u_right)
        expected = x  # Linear interpolation

        assert jnp.allclose(u, expected, atol=1e-10)

    def test_steady_state_boundaries(self):
        """Steady state should match boundary values."""
        x = jnp.linspace(0, 1, 100)
        u_left, u_right = 2.5, 7.3

        u = steady_state_1d(x, u_left, u_right)

        assert jnp.isclose(u[0], u_left)
        assert jnp.isclose(u[-1], u_right)


class TestPDEResidual:
    """Test PDE residual computation."""

    def test_residual_steady_state(self):
        """Residual should be zero for steady-state solution."""
        # For steady state, u_t = 0 and u_xx = 0
        # So residual = u_t - alpha * u_xx = 0

        def u_fn(params, x, t):
            # Linear steady state: u(x) = x
            return x

        x = jnp.linspace(0.1, 0.9, 10)[:, None]
        t = jnp.ones_like(x) * 0.5

        residual = heat_equation_residual(u_fn, {}, x, t, alpha=0.01)

        # Residual should be near zero (numerical precision)
        assert jnp.allclose(residual, 0.0, atol=1e-5)

    def test_residual_shape(self):
        """Residual should have same shape as input points."""

        def u_fn(params, x, t):
            return jnp.sin(x) * jnp.exp(-t)

        n_points = 50
        x = jnp.linspace(0, 1, n_points)[:, None]
        t = jnp.linspace(0, 1, n_points)[:, None]

        residual = heat_equation_residual(u_fn, {}, x, t)

        assert residual.shape == (n_points,)

    @pytest.mark.parametrize("alpha", [0.001, 0.01, 0.1, 1.0])
    def test_residual_alpha_dependence(self, alpha):
        """Residual should scale with diffusivity alpha."""

        def u_fn(params, x, t):
            # Simple polynomial: u = x^2 * t
            return x**2 * t

        x = jnp.array([[0.5]])
        t = jnp.array([[0.5]])

        residual = heat_equation_residual(u_fn, {}, x, t, alpha=alpha)

        # For u = x^2 * t:
        # u_t = x^2
        # u_xx = 2t
        # residual = x^2 - alpha * 2t = 0.25 - alpha * 1.0
        expected = 0.25 - alpha * 1.0

        assert jnp.allclose(residual, expected, rtol=1e-4)


class TestBoundaryConditions:
    """Test boundary condition enforcement."""

    def test_dirichlet_bc(self):
        """Dirichlet BC should be exactly satisfied."""
        from phio.physics.heat import dirichlet_bc_loss

        def u_fn(params, x, t):
            # Perfect BC satisfaction: u(0, t) = 0, u(1, t) = 1
            return x

        x_bc = jnp.array([[0.0], [1.0]])
        t_bc = jnp.array([[0.5], [0.5]])
        u_bc = jnp.array([[0.0], [1.0]])

        loss = dirichlet_bc_loss(u_fn, {}, x_bc, t_bc, u_bc)

        assert jnp.allclose(loss, 0.0, atol=1e-10)

    def test_initial_condition(self):
        """Initial condition should be exactly satisfied."""
        from phio.physics.heat import initial_condition_loss

        def u_fn(params, x, t):
            # u(x, 0) = sin(pi * x)
            return jnp.sin(jnp.pi * x)

        x_ic = jnp.linspace(0, 1, 50)[:, None]
        u_ic = jnp.sin(jnp.pi * x_ic.flatten())[:, None]

        loss = initial_condition_loss(u_fn, {}, x_ic, u_ic)

        assert jnp.allclose(loss, 0.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
