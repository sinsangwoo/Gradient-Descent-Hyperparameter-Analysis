"""Navier-Stokes PINN trainer with velocity-pressure formulation.

Trains a neural network to solve 2D incompressible Navier-Stokes equations.
"""

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm


class NSTrainState(train_state.TrainState):
    """Extended train state for Navier-Stokes with loss weights."""

    momentum_weight: float = 1.0
    continuity_weight: float = 1.0
    bc_weight: float = 1.0
    ic_weight: float = 1.0


def create_ns_train_state(
    rng: jax.random.PRNGKey,
    model: Callable,
    learning_rate: float = 1e-3,
    sample_input: Optional[Tuple[jnp.ndarray, ...]] = None,
) -> NSTrainState:
    """Initialize training state for Navier-Stokes PINN.

    Args:
        rng: Random key
        model: Flax model outputting [u, v, p]
        learning_rate: Adam learning rate
        sample_input: (x, y, t) for parameter initialization

    Returns:
        Initialized NSTrainState
    """
    if sample_input is None:
        x_sample = jnp.ones((1, 1))
        y_sample = jnp.ones((1, 1))
        t_sample = jnp.zeros((1, 1))
    else:
        x_sample, y_sample, t_sample = sample_input

    params = model.init(rng, x_sample, y_sample, t_sample)
    tx = optax.adam(learning_rate)

    return NSTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def compute_ns_loss(
    params: dict,
    apply_fn: Callable,
    x_pde: jnp.ndarray,
    y_pde: jnp.ndarray,
    t_pde: jnp.ndarray,
    x_bc: jnp.ndarray,
    y_bc: jnp.ndarray,
    t_bc: jnp.ndarray,
    u_bc: jnp.ndarray,
    v_bc: jnp.ndarray,
    x_ic: jnp.ndarray,
    y_ic: jnp.ndarray,
    u_ic: jnp.ndarray,
    v_ic: jnp.ndarray,
    ns_residual_fn: Callable,
    nu: float = 0.01,
    momentum_weight: float = 1.0,
    continuity_weight: float = 1.0,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute total Navier-Stokes PINN loss.

    Loss = momentum + continuity + BC + IC

    Args:
        params: Model parameters
        apply_fn: Model application function
        x_pde, y_pde, t_pde: PDE collocation points
        x_bc, y_bc, t_bc, u_bc, v_bc: Boundary conditions
        x_ic, y_ic, u_ic, v_ic: Initial conditions
        ns_residual_fn: Navier-Stokes residual function
        nu: Kinematic viscosity
        *_weight: Loss term weights

    Returns:
        total_loss: Weighted sum of losses
        loss_dict: Individual loss components
    """

    def predict_single(params_inner, x, y, t):
        return apply_fn(params_inner, x[None, :], y[None, :], t[None, :])[0]

    # PDE residuals
    r_u, r_v, r_cont = jax.vmap(
        lambda x, y, t: ns_residual_fn(predict_single, params, x, y, t, nu=nu)
    )(x_pde, y_pde, t_pde)

    loss_momentum = jnp.mean(r_u**2 + r_v**2)
    loss_continuity = jnp.mean(r_cont**2)

    # Boundary conditions
    uvp_bc_pred = jax.vmap(apply_fn, in_axes=(None, 0, 0, 0))(
        params, x_bc[:, None], y_bc[:, None], t_bc[:, None]
    )
    u_bc_pred = uvp_bc_pred[:, 0]
    v_bc_pred = uvp_bc_pred[:, 1]

    loss_bc = jnp.mean((u_bc_pred - u_bc) ** 2 + (v_bc_pred - v_bc) ** 2)

    # Initial conditions
    t_ic = jnp.zeros_like(x_ic)
    uvp_ic_pred = jax.vmap(apply_fn, in_axes=(None, 0, 0, 0))(
        params, x_ic[:, None], y_ic[:, None], t_ic[:, None]
    )
    u_ic_pred = uvp_ic_pred[:, 0]
    v_ic_pred = uvp_ic_pred[:, 1]

    loss_ic = jnp.mean((u_ic_pred - u_ic) ** 2 + (v_ic_pred - v_ic) ** 2)

    # Total loss
    total_loss = (
        momentum_weight * loss_momentum
        + continuity_weight * loss_continuity
        + bc_weight * loss_bc
        + ic_weight * loss_ic
    )

    loss_dict = {
        "total": total_loss,
        "momentum": loss_momentum,
        "continuity": loss_continuity,
        "bc": loss_bc,
        "ic": loss_ic,
    }

    return total_loss, loss_dict


@partial(jax.jit, static_argnames=["apply_fn", "tx"])
def train_step_jitted(
    params: dict,
    apply_fn: Callable,
    opt_state: optax.OptState,
    tx: optax.GradientTransformation,
    x_pde: jnp.ndarray,
    y_pde: jnp.ndarray,
    t_pde: jnp.ndarray,
    x_bc: jnp.ndarray,
    y_bc: jnp.ndarray,
    t_bc: jnp.ndarray,
    u_bc: jnp.ndarray,
    v_bc: jnp.ndarray,
    x_ic: jnp.ndarray,
    y_ic: jnp.ndarray,
    u_ic: jnp.ndarray,
    v_ic: jnp.ndarray,
    nu: float,
    momentum_weight: float,
    continuity_weight: float,
    bc_weight: float,
    ic_weight: float,
) -> Tuple[dict, optax.OptState, Dict[str, jnp.ndarray]]:
    """JIT-compiled training step."""

    def loss_fn(p):
        from phio.physics.navier_stokes import ns_residual_2d

        return compute_ns_loss(
            p,
            apply_fn,
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
            ns_residual_2d,
            nu=nu,
            momentum_weight=momentum_weight,
            continuity_weight=continuity_weight,
            bc_weight=bc_weight,
            ic_weight=ic_weight,
        )

    (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss_dict


def train_ns_pinn(
    state: NSTrainState,
    x_pde: jnp.ndarray,
    y_pde: jnp.ndarray,
    t_pde: jnp.ndarray,
    x_bc: jnp.ndarray,
    y_bc: jnp.ndarray,
    t_bc: jnp.ndarray,
    u_bc: jnp.ndarray,
    v_bc: jnp.ndarray,
    x_ic: jnp.ndarray,
    y_ic: jnp.ndarray,
    u_ic: jnp.ndarray,
    v_ic: jnp.ndarray,
    nu: float = 0.01,
    num_epochs: int = 10000,
    print_every: int = 1000,
) -> Tuple[NSTrainState, Dict[str, list]]:
    """Train Navier-Stokes PINN.

    Args:
        state: Initial training state
        (data args): Training data
        nu: Kinematic viscosity
        num_epochs: Number of training epochs
        print_every: Print frequency

    Returns:
        trained_state: Final training state
        history: Training history
    """
    history = {"total": [], "momentum": [], "continuity": [], "bc": [], "ic": []}

    for epoch in tqdm(range(num_epochs), desc="Training NS-PINN"):
        # Training step
        new_params, new_opt_state, loss_dict = train_step_jitted(
            state.params,
            state.apply_fn,
            state.opt_state,
            state.tx,
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
            nu,
            state.momentum_weight,
            state.continuity_weight,
            state.bc_weight,
            state.ic_weight,
        )

        state = state.replace(params=new_params, opt_state=new_opt_state)

        # Log history
        for key in history:
            history[key].append(float(loss_dict[key]))

        # Print progress
        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Loss: {loss_dict['total']:.6f} | "
                f"Momentum: {loss_dict['momentum']:.6f} | "
                f"Continuity: {loss_dict['continuity']:.6f}"
            )

    return state, history
