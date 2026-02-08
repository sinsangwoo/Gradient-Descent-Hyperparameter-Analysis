"""2D incompressible Navier-Stokes equation for PINN.

Implements momentum and continuity equations:
    u_t + u*u_x + v*u_y = -p_x + nu*(u_xx + u_yy)  [x-momentum]
    v_t + u*v_x + v*v_y = -p_y + nu*(v_xx + v_yy)  [y-momentum]
    u_x + v_y = 0                                  [continuity]

where:
    u, v: velocity components (x, y)
    p: pressure
    nu: kinematic viscosity
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp


def ns_residual_2d(
    u_fn: Callable,
    params: dict,
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
    nu: float = 0.01,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Navier-Stokes residual for 2D incompressible flow.

    Args:
        u_fn: Neural network function(params, x, y, t) -> [u, v, p]
        params: Network parameters
        x: Spatial coordinate (horizontal)
        y: Spatial coordinate (vertical)
        t: Time coordinate
        nu: Kinematic viscosity

    Returns:
        r_u: Residual of x-momentum equation
        r_v: Residual of y-momentum equation
        r_cont: Residual of continuity equation
    """

    def network_output(x_val, y_val, t_val):
        """Evaluate network at a point."""
        output = u_fn(params, x_val, y_val, t_val)
        return output[0], output[1], output[2]  # u, v, p

    # First derivatives
    u, v, p = network_output(x, y, t)

    # Velocity gradients
    u_x = jax.grad(lambda x_: network_output(x_, y, t)[0])(x)
    u_y = jax.grad(lambda y_: network_output(x, y_, t)[0])(y)
    u_t = jax.grad(lambda t_: network_output(x, y, t_)[0])(t)

    v_x = jax.grad(lambda x_: network_output(x_, y, t)[1])(x)
    v_y = jax.grad(lambda y_: network_output(x, y_, t)[1])(y)
    v_t = jax.grad(lambda t_: network_output(x, y, t_)[1])(t)

    # Pressure gradients
    p_x = jax.grad(lambda x_: network_output(x_, y, t)[2])(x)
    p_y = jax.grad(lambda y_: network_output(x, y_, t)[2])(y)

    # Second derivatives (Laplacian terms)
    u_xx = jax.grad(jax.grad(lambda x_: network_output(x_, y, t)[0]))(x)
    u_yy = jax.grad(jax.grad(lambda y_: network_output(x, y_, t)[0]))(y)

    v_xx = jax.grad(jax.grad(lambda x_: network_output(x_, y, t)[1]))(x)
    v_yy = jax.grad(jax.grad(lambda y_: network_output(x, y_, t)[1]))(y)

    # Momentum equations
    r_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    r_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    # Continuity equation (incompressibility)
    r_cont = u_x + v_y

    return r_u, r_v, r_cont


def lid_driven_cavity_bc(
    x: jnp.ndarray,
    y: jnp.ndarray,
    u_lid: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Boundary conditions for lid-driven cavity.

    Domain: [0,1] x [0,1]
    BC:
        - Top (y=1): u = u_lid, v = 0 (moving lid)
        - Others: u = 0, v = 0 (no-slip walls)

    Args:
        x: x-coordinates on boundary
        y: y-coordinates on boundary
        u_lid: Lid velocity (default: 1.0)

    Returns:
        u_bc: x-velocity boundary values
        v_bc: y-velocity boundary values
    """
    # Top wall (y = 1): moving lid
    is_top = jnp.isclose(y, 1.0, atol=1e-6)
    u_bc = jnp.where(is_top, u_lid, 0.0)
    v_bc = jnp.zeros_like(x)

    return u_bc, v_bc


def analytical_taylor_green(
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
    nu: float = 0.01,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Taylor-Green vortex analytical solution.

    Exact solution for testing PINN accuracy:
        u = -cos(x) * sin(y) * exp(-2*nu*t)
        v = sin(x) * cos(y) * exp(-2*nu*t)
        p = -0.25 * (cos(2x) + cos(2y)) * exp(-4*nu*t)

    Args:
        x, y: Spatial coordinates
        t: Time
        nu: Kinematic viscosity

    Returns:
        u, v, p: Velocity and pressure fields
    """
    decay = jnp.exp(-2.0 * nu * t)
    u = -jnp.cos(x) * jnp.sin(y) * decay
    v = jnp.sin(x) * jnp.cos(y) * decay
    p = -0.25 * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y)) * jnp.exp(-4.0 * nu * t)

    return u, v, p
