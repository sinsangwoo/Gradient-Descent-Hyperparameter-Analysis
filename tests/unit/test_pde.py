"""Test PDE base class and implementations."""

import pytest
import jax.numpy as jnp
from phio.physics import HeatEquation1D, WaveEquation1D


class TestHeatEquation1D:
    """Test suite for 1D heat equation."""

    def test_initialization(self):
        """Test basic initialization."""
        pde = HeatEquation1D(domain=(0, 1), diffusion_coeff=0.01)
        assert pde.domain == (0, 1)
        assert pde.alpha == 0.01

    def test_exact_solution_shape(self, sample_1d_grid):
        """Test exact solution has correct shape."""
        x, t = sample_1d_grid
        pde = HeatEquation1D()
        
        # Test at single point
        u = pde.exact_solution(x[0], t[0])
        assert isinstance(u, jnp.ndarray)
        
        # Test at all grid points
        X, T = jnp.meshgrid(x, t)
        u_grid = pde.exact_solution(X, T)
        assert u_grid.shape == X.shape

    def test_exact_solution_boundary_conditions(self):
        """Test exact solution satisfies boundary conditions."""
        pde = HeatEquation1D(domain=(0, 1))
        t_test = jnp.linspace(0, 1, 100)
        
        # u(0, t) = 0
        u_left = pde.exact_solution(jnp.zeros_like(t_test), t_test)
        assert jnp.allclose(u_left, 0.0, atol=1e-10)
        
        # u(1, t) = 0
        u_right = pde.exact_solution(jnp.ones_like(t_test), t_test)
        assert jnp.allclose(u_right, 0.0, atol=1e-10)

    def test_exact_solution_initial_condition(self):
        """Test exact solution satisfies initial condition."""
        pde = HeatEquation1D()
        x_test = jnp.linspace(0, 1, 100)
        
        # u(x, 0) = sin(pi * x)
        u_initial = pde.exact_solution(x_test, jnp.zeros_like(x_test))
        expected = jnp.sin(jnp.pi * x_test)
        assert jnp.allclose(u_initial, expected, atol=1e-10)


class TestWaveEquation1D:
    """Test suite for 1D wave equation."""

    def test_initialization(self):
        """Test basic initialization."""
        pde = WaveEquation1D(domain=(0, 1), wave_speed=1.0)
        assert pde.domain == (0, 1)
        assert pde.c == 1.0

    def test_exact_solution_periodicity(self):
        """Test exact solution has correct temporal periodicity."""
        pde = WaveEquation1D(wave_speed=1.0)
        x = jnp.array([0.5])
        
        # Wave should be periodic with period T = 2/c = 2
        u_t0 = pde.exact_solution(x, jnp.array([0.0]))
        u_t2 = pde.exact_solution(x, jnp.array([2.0]))
        assert jnp.allclose(u_t0, u_t2, atol=1e-6)
