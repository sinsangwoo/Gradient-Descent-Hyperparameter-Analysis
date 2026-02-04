"""Heat equation PDE definitions and analytical solutions.

This module provides:
- Heat equation residual computation using JAX autodiff
- Analytical solutions for validation (1D Gaussian, steady-state)
- Boundary condition enforcement utilities
"""

from typing import Callable

import jax
import jax.numpy as jnp


def heat_equation_residual(
    u_fn: Callable,
    params: dict,
    x: jnp.ndarray,
    t: jnp.ndarray,
    alpha: float = 0.01,
) -> jnp.ndarray:
    """Compute residual of 1D heat equation: ∂u/∂t = α ∂²u/∂x².

    Args:
        u_fn: Neural network function u(params, x, t) -> u(x, t)
        params: Neural network parameters (Flax pytree)
        x: Spatial coordinates, shape (N, 1)
        t: Time coordinates, shape (N, 1)
        alpha: Thermal diffusivity coefficient (default: 0.01)

    Returns:
        PDE residual at each (x, t) point, shape (N,)

    Example:
        >>> residual = heat_equation_residual(model.apply, params, x, t, alpha=0.01)
        >>> pde_loss = jnp.mean(residual**2)
    """

    # Compute derivatives using JAX autodiff
    def u_t_fn(t_val, x_val):
        return u_fn(params, x_val, t_val)

    def u_x_fn(x_val, t_val):
        return u_fn(params, x_val, t_val)

    # First derivatives
    u_t = jax.vmap(jax.grad(u_t_fn, argnums=0))(t.flatten(), x.flatten())

    # Second derivative in space
    u_xx = jax.vmap(jax.grad(jax.grad(u_x_fn, argnums=0), argnums=0))(
        x.flatten(), t.flatten()
    )

    # Heat equation residual: u_t - alpha * u_xx = 0
    residual = u_t - alpha * u_xx

    return residual


def analytical_gaussian(
    x: jnp.ndarray,
    t: jnp.ndarray,
    alpha: float = 0.01,
    x0: float = 0.5,
    sigma0: float = 0.1,
) -> jnp.ndarray:
    """Analytical solution for 1D heat equation with Gaussian initial condition.

    Initial condition: u(x, 0) = exp(-(x - x0)² / (2σ₀²))
    Solution: u(x, t) = (σ₀ / σ(t)) * exp(-(x - x0)² / (2σ(t)²))
    where σ(t)² = σ₀² + 2αt

    Args:
        x: Spatial coordinates, shape (N,) or (N, 1)
        t: Time coordinates, shape (N,) or (N, 1)
        alpha: Thermal diffusivity
        x0: Initial peak location
        sigma0: Initial standard deviation

    Returns:
        Temperature u(x, t), shape matching broadcast of (x, t)
    """
    x = jnp.atleast_1d(x).flatten()
    t = jnp.atleast_1d(t).flatten()

    # Time-dependent variance
    sigma_t_sq = sigma0**2 + 2 * alpha * t[:, None]

    # Gaussian solution
    u = (sigma0 / jnp.sqrt(sigma_t_sq)) * jnp.exp(
        -(x[None, :] - x0) ** 2 / (2 * sigma_t_sq)
    )

    return u


def steady_state_1d(
    x: jnp.ndarray,
    u_left: float = 0.0,
    u_right: float = 1.0,
) -> jnp.ndarray:
    """Analytical steady-state solution for 1D heat equation.

    Boundary conditions: u(0, t) = u_left, u(1, t) = u_right
    Steady state: ∂u/∂t = 0 → ∂²u/∂x² = 0 → u(x) = u_left + (u_right - u_left) * x

    Args:
        x: Spatial coordinates in [0, 1], shape (N,) or (N, 1)
        u_left: Temperature at x=0
        u_right: Temperature at x=1

    Returns:
        Temperature u(x), shape matching x
    """
    x = jnp.atleast_1d(x).flatten()
    return u_left + (u_right - u_left) * x


def dirichlet_bc_loss(
    u_fn: Callable,
    params: dict,
    x_bc: jnp.ndarray,
    t_bc: jnp.ndarray,
    u_bc: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Dirichlet boundary condition loss.

    Args:
        u_fn: Neural network function
        params: Network parameters
        x_bc: Boundary spatial coordinates, shape (N_bc, 1)
        t_bc: Boundary time coordinates, shape (N_bc, 1)
        u_bc: Target boundary values, shape (N_bc, 1)

    Returns:
        Boundary condition loss (MSE)
    """
    u_pred = jax.vmap(u_fn, in_axes=(None, 0, 0))(params, x_bc, t_bc)
    return jnp.mean((u_pred - u_bc) ** 2)


def initial_condition_loss(
    u_fn: Callable,
    params: dict,
    x_ic: jnp.ndarray,
    u_ic: jnp.ndarray,
) -> jnp.ndarray:
    """Compute initial condition loss.

    Args:
        u_fn: Neural network function
        params: Network parameters
        x_ic: Initial spatial coordinates, shape (N_ic, 1)
        u_ic: Initial temperature distribution, shape (N_ic, 1)

    Returns:
        Initial condition loss (MSE)
    """
    t_ic = jnp.zeros_like(x_ic)
    u_pred = jax.vmap(u_fn, in_axes=(None, 0, 0))(params, x_ic, t_ic)
    return jnp.mean((u_pred - u_ic) ** 2)
