"""Physics-informed loss functions."""

import jax
import jax.numpy as jnp
from typing import Callable, Dict


def compute_pde_residual(
    network_fn: Callable,
    params: dict,
    x: jnp.ndarray,
    t: jnp.ndarray,
    pde_type: str = "heat",
    alpha: float = 0.01,
) -> jnp.ndarray:
    """Compute PDE residual using automatic differentiation.

    For heat equation: u_t - alpha * u_xx = 0
    For wave equation: u_tt - c^2 * u_xx = 0

    Args:
        network_fn: Neural network forward function
        params: Network parameters
        x: Spatial coordinates
        t: Temporal coordinates
        pde_type: Type of PDE ('heat', 'wave')
        alpha: PDE coefficient (diffusion or wave speed)

    Returns:
        PDE residual at each point (should be ~0)
    """

    def u_fn(x_val, t_val):
        """Evaluate network at single point."""
        inputs = jnp.array([x_val, t_val]).reshape(1, -1)
        return network_fn(params, inputs).squeeze()

    if pde_type == "heat":
        # Heat equation: u_t = alpha * u_xx
        # Compute derivatives
        u_t = jax.grad(u_fn, argnums=1)  # ∂u/∂t
        u_x = jax.grad(u_fn, argnums=0)  # ∂u/∂x
        u_xx = jax.grad(u_x, argnums=0)  # ∂²u/∂x²

        # Vectorize over batch
        u_t_batch = jax.vmap(lambda xi, ti: u_t(xi, ti))(x, t)
        u_xx_batch = jax.vmap(lambda xi, ti: u_xx(xi, ti))(x, t)

        # Residual: u_t - alpha * u_xx
        residual = u_t_batch - alpha * u_xx_batch

    elif pde_type == "wave":
        # Wave equation: u_tt = c^2 * u_xx
        u_t = jax.grad(u_fn, argnums=1)
        u_tt = jax.grad(u_t, argnums=1)  # ∂²u/∂t²
        u_x = jax.grad(u_fn, argnums=0)
        u_xx = jax.grad(u_x, argnums=0)  # ∂²u/∂x²

        u_tt_batch = jax.vmap(lambda xi, ti: u_tt(xi, ti))(x, t)
        u_xx_batch = jax.vmap(lambda xi, ti: u_xx(xi, ti))(x, t)

        # Residual: u_tt - c^2 * u_xx
        residual = u_tt_batch - alpha**2 * u_xx_batch

    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")

    return residual


def pinn_loss(
    params: dict,
    network_fn: Callable,
    x_colloc: jnp.ndarray,
    t_colloc: jnp.ndarray,
    x_bc: jnp.ndarray,
    t_bc: jnp.ndarray,
    u_bc: jnp.ndarray,
    x_ic: jnp.ndarray,
    t_ic: jnp.ndarray,
    u_ic: jnp.ndarray,
    weights: Dict[str, float] = None,
    pde_type: str = "heat",
    alpha: float = 0.01,
) -> jnp.ndarray:
    """Compute total PINN loss.

    Loss = w_pde * L_pde + w_bc * L_bc + w_ic * L_ic

    Args:
        params: Network parameters
        network_fn: Forward function
        x_colloc, t_colloc: Collocation points for PDE residual
        x_bc, t_bc, u_bc: Boundary condition points and values
        x_ic, t_ic, u_ic: Initial condition points and values
        weights: Loss component weights {'pde': 1.0, 'bc': 1.0, 'ic': 1.0}
        pde_type: Type of PDE
        alpha: PDE coefficient

    Returns:
        Total loss (scalar)
    """
    if weights is None:
        weights = {"pde": 1.0, "bc": 1.0, "ic": 1.0}

    # PDE residual loss
    residual = compute_pde_residual(
        network_fn, params, x_colloc, t_colloc, pde_type, alpha
    )
    loss_pde = jnp.mean(residual**2)

    # Boundary condition loss
    inputs_bc = jnp.stack([x_bc, t_bc], axis=-1)
    u_pred_bc = network_fn(params, inputs_bc).squeeze()
    loss_bc = jnp.mean((u_pred_bc - u_bc) ** 2)

    # Initial condition loss
    inputs_ic = jnp.stack([x_ic, t_ic], axis=-1)
    u_pred_ic = network_fn(params, inputs_ic).squeeze()
    loss_ic = jnp.mean((u_pred_ic - u_ic) ** 2)

    # Weighted total loss
    total_loss = (
        weights["pde"] * loss_pde + weights["bc"] * loss_bc + weights["ic"] * loss_ic
    )

    return total_loss
