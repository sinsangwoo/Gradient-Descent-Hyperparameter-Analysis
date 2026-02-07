"""Unit tests for Navier-Stokes physics module."""

import jax
import jax.numpy as jnp
import pytest

from phio.physics.navier_stokes import (
    analytical_taylor_green,
    lid_driven_cavity_bc,
    ns_residual_2d,
)


class TestNSResidual:
    """Test Navier-Stokes residual computation."""

    def test_taylor_green_satisfies_ns(self):
        """Taylor-Green vortex should satisfy NS equations."""

        def taylor_green_fn(params, x, y, t):
            """Analytical Taylor-Green solution."""
            u, v, p = analytical_taylor_green(x, y, t, nu=0.01)
            return jnp.array([u, v, p])

        x, y, t = 0.5, 0.5, 0.1
        params = {}  # Analytical solution doesn't need parameters

        r_u, r_v, r_cont = ns_residual_2d(taylor_green_fn, params, x, y, t, nu=0.01)

        # Residuals should be near zero for exact solution
        assert jnp.abs(r_u) < 1e-5
        assert jnp.abs(r_v) < 1e-5
        assert jnp.abs(r_cont) < 1e-5

    def test_residual_shape(self):
        """Residual should return scalar values."""

        def dummy_fn(params, x, y, t):
            return jnp.array([x + y, x - y, x * y])

        x, y, t = 0.5, 0.5, 0.0
        params = {}

        r_u, r_v, r_cont = ns_residual_2d(dummy_fn, params, x, y, t)

        assert isinstance(r_u, jnp.ndarray)
        assert isinstance(r_v, jnp.ndarray)
        assert isinstance(r_cont, jnp.ndarray)
        assert r_u.shape == ()
        assert r_v.shape == ()
        assert r_cont.shape == ()


class TestBoundaryConditions:
    """Test boundary condition functions."""

    def test_lid_driven_cavity_top_wall(self):
        """Top wall should have u=u_lid, v=0."""
        x = jnp.array([0.0, 0.5, 1.0])
        y = jnp.array([1.0, 1.0, 1.0])  # Top wall

        u_bc, v_bc = lid_driven_cavity_bc(x, y, u_lid=1.0)

        assert jnp.allclose(u_bc, 1.0)
        assert jnp.allclose(v_bc, 0.0)

    def test_lid_driven_cavity_other_walls(self):
        """Other walls should have u=0, v=0 (no-slip)."""
        # Bottom wall
        x_bottom = jnp.array([0.0, 0.5, 1.0])
        y_bottom = jnp.array([0.0, 0.0, 0.0])

        u_bc, v_bc = lid_driven_cavity_bc(x_bottom, y_bottom)

        assert jnp.allclose(u_bc, 0.0)
        assert jnp.allclose(v_bc, 0.0)

        # Left wall
        x_left = jnp.array([0.0, 0.0, 0.0])
        y_left = jnp.array([0.0, 0.5, 1.0])

        u_bc, v_bc = lid_driven_cavity_bc(x_left, y_left)

        assert jnp.allclose(u_bc[:-1], 0.0)  # Except corner (y=1)
        assert jnp.allclose(v_bc, 0.0)


class TestAnalyticalSolutions:
    """Test analytical solution implementations."""

    def test_taylor_green_initial_condition(self):
        """At t=0, Taylor-Green should match initial state."""
        x = jnp.linspace(0, 2 * jnp.pi, 10)
        y = jnp.linspace(0, 2 * jnp.pi, 10)
        x_grid, y_grid = jnp.meshgrid(x, y)

        u, v, p = analytical_taylor_green(x_grid, y_grid, 0.0, nu=0.01)

        # At t=0, exp(-2*nu*t) = 1
        u_expected = -jnp.cos(x_grid) * jnp.sin(y_grid)
        v_expected = jnp.sin(x_grid) * jnp.cos(y_grid)

        assert jnp.allclose(u, u_expected)
        assert jnp.allclose(v, v_expected)

    def test_taylor_green_decay(self):
        """Taylor-Green should decay exponentially with time."""
        x, y = 1.0, 1.0
        t1, t2 = 0.0, 1.0
        nu = 0.01

        u1, v1, p1 = analytical_taylor_green(x, y, t1, nu=nu)
        u2, v2, p2 = analytical_taylor_green(x, y, t2, nu=nu)

        # Velocity should decay
        assert jnp.abs(u2) < jnp.abs(u1)
        assert jnp.abs(v2) < jnp.abs(v1)

        # Decay factor should be exp(-2*nu*t)
        expected_decay = jnp.exp(-2.0 * nu * t2) / jnp.exp(-2.0 * nu * t1)
        actual_decay = u2 / u1 if u1 != 0 else 0

        assert jnp.abs(actual_decay - expected_decay) < 1e-6

    def test_taylor_green_satisfies_continuity(self):
        """Taylor-Green should satisfy div(u) = 0."""
        x, y, t = 1.0, 1.0, 0.5
        nu = 0.01

        # Compute divergence numerically
        eps = 1e-6
        u_x_plus, _, _ = analytical_taylor_green(x + eps, y, t, nu)
        u_x_minus, _, _ = analytical_taylor_green(x - eps, y, t, nu)
        u_x = (u_x_plus - u_x_minus) / (2 * eps)

        _, v_y_plus, _ = analytical_taylor_green(x, y + eps, t, nu)
        _, v_y_minus, _ = analytical_taylor_green(x, y - eps, t, nu)
        v_y = (v_y_plus - v_y_minus) / (2 * eps)

        divergence = u_x + v_y

        assert jnp.abs(divergence) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
