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

    Uses JAX automatic differentiation with jacrev for efficient batch computation.

    Args:
        u_fn: Neural network function(params, x, y, t) -> [u, v, p]
        params: Network parameters
        x: Spatial coordinate (horizontal), shape (N,)
        y: Spatial coordinate (vertical), shape (N,)
        t: Time coordinate, shape (N,)
        nu: Kinematic viscosity

    Returns:
        r_u: Residual of x-momentum equation, shape (N,)
        r_v: Residual of y-momentum equation, shape (N,)
        r_cont: Residual of continuity equation, shape (N,)
    """

    def net_fn(coords):
        """Network wrapper for differentiation.
        
        Args:
            coords: [x, y, t] coordinates
            
        Returns:
            [u, v, p] output
        """
        return u_fn(params, coords[0], coords[1], coords[2])

    # Jacobian function: computes first derivatives
    # Output shape: [3, 3] for [u, v, p] w.r.t [x, y, t]
    jac_fn = jax.jacrev(net_fn)

    # Hessian functions: computes second derivatives
    hess_u_fn = jax.jacrev(lambda c: jac_fn(c)[0, :])  # u derivatives
    hess_v_fn = jax.jacrev(lambda c: jac_fn(c)[1, :])  # v derivatives

    def single_point_residual(coords):
        """Compute residual at a single collocation point.
        
        Args:
            coords: [x, y, t] for single point
            
        Returns:
            (res_u, res_v, res_c): Residuals for momentum and continuity
        """
        # Network output
        out = net_fn(coords)
        u, v, p = out[0], out[1], out[2]

        # First derivatives via Jacobian
        # jac = [[u_x, u_y, u_t],
        #        [v_x, v_y, v_t],
        #        [p_x, p_y, p_t]]
        jac = jac_fn(coords)
        u_x, u_y, u_t = jac[0, 0], jac[0, 1], jac[0, 2]
        v_x, v_y, v_t = jac[1, 0], jac[1, 1], jac[1, 2]
        p_x, p_y = jac[2, 0], jac[2, 1]

        # Second derivatives via Hessian
        u_xx = hess_u_fn(coords)[0, 0]
        u_yy = hess_u_fn(coords)[1, 1]
        v_xx = hess_v_fn(coords)[0, 0]
        v_yy = hess_v_fn(coords)[1, 1]

        # Navier-Stokes residuals
        res_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        res_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        res_c = u_x + v_y  # Continuity

        return res_u, res_v, res_c

    # Vectorize over batch using vmap
    coords_batch = jnp.stack([x, y, t], axis=-1)  # Shape: (N, 3)
    r_u, r_v, r_cont = jax.vmap(single_point_residual)(coords_batch)

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
