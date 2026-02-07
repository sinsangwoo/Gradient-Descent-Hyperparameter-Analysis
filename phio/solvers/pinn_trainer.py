"""PINN training loop with curriculum learning and adaptive weighting."""

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm


class TrainState(train_state.TrainState):
    """Extended train state with loss weights for curriculum learning."""

    pde_weight: float = 1.0
    bc_weight: float = 1.0
    ic_weight: float = 1.0


def create_train_state(
    rng: jax.random.PRNGKey,
    model: Callable,
    learning_rate: float = 1e-3,
    sample_input: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
) -> TrainState:
    """Initialize training state.

    Args:
        rng: Random key
        model: Flax model
        learning_rate: Adam learning rate
        sample_input: (x_sample, t_sample) for parameter initialization

    Returns:
        Initialized TrainState
    """
    if sample_input is None:
        x_sample = jnp.ones((1, 1))
        t_sample = jnp.zeros((1, 1))
    else:
        x_sample, t_sample = sample_input

    params = model.init(rng, x_sample, t_sample)
    tx = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def compute_pinn_loss(
    params: dict,
    apply_fn: Callable,
    x_pde: jnp.ndarray,
    t_pde: jnp.ndarray,
    x_bc: jnp.ndarray,
    t_bc: jnp.ndarray,
    u_bc: jnp.ndarray,
    x_ic: jnp.ndarray,
    u_ic: jnp.ndarray,
    pde_residual_fn: Callable,
    alpha: float = 0.01,
    pde_weight: float = 1.0,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute total PINN loss with components.

    Loss = pde_weight * L_pde + bc_weight * L_bc + ic_weight * L_ic

    Args:
        params: Model parameters
        apply_fn: Model application function
        x_pde: PDE collocation points (space)
        t_pde: PDE collocation points (time)
        x_bc, t_bc, u_bc: Boundary condition data
        x_ic, u_ic: Initial condition data
        pde_residual_fn: Function to compute PDE residual
        alpha: Thermal diffusivity
        pde_weight, bc_weight, ic_weight: Loss term weights

    Returns:
        total_loss: Weighted sum of losses
        loss_dict: Individual loss components
    """

    # PDE residual loss - properly vectorize over all collocation points
    def predict_single(params_inner, x, t):
        return apply_fn(params_inner, x[None, :], t[None, :])[0]

    # Vectorize residual computation
    residual = jax.vmap(
        lambda x, t: pde_residual_fn(predict_single, params, x, t, alpha=alpha)
    )(x_pde, t_pde)
    loss_pde = jnp.mean(residual**2)

    # Boundary condition loss
    u_bc_pred = jax.vmap(apply_fn, in_axes=(None, 0, 0))(
        params, x_bc[:, None], t_bc[:, None]
    ).squeeze()
    loss_bc = jnp.mean((u_bc_pred - u_bc) ** 2)

    # Initial condition loss
    t_ic = jnp.zeros_like(x_ic)
    u_ic_pred = jax.vmap(apply_fn, in_axes=(None, 0, 0))(
        params, x_ic[:, None], t_ic[:, None]
    ).squeeze()
    loss_ic = jnp.mean((u_ic_pred - u_ic) ** 2)

    # Total loss
    total_loss = pde_weight * loss_pde + bc_weight * loss_bc + ic_weight * loss_ic

    loss_dict = {
        "total": total_loss,
        "pde": loss_pde,
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
    t_pde: jnp.ndarray,
    x_bc: jnp.ndarray,
    t_bc: jnp.ndarray,
    u_bc: jnp.ndarray,
    x_ic: jnp.ndarray,
    u_ic: jnp.ndarray,
    alpha: float,
    pde_weight: float,
    bc_weight: float,
    ic_weight: float,
) -> Tuple[dict, optax.OptState, Dict[str, jnp.ndarray]]:
    """JIT-compiled training step with static function arguments."""

    def loss_fn(p):
        # Use heat_equation_residual directly
        from phio.physics.heat import heat_equation_residual

        return compute_pinn_loss(
            p,
            apply_fn,
            x_pde,
            t_pde,
            x_bc,
            t_bc,
            u_bc,
            x_ic,
            u_ic,
            heat_equation_residual,
            alpha=alpha,
            pde_weight=pde_weight,
            bc_weight=bc_weight,
            ic_weight=ic_weight,
        )

    (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss_dict


def train_step(
    state: TrainState,
    x_pde: jnp.ndarray,
    t_pde: jnp.ndarray,
    x_bc: jnp.ndarray,
    t_bc: jnp.ndarray,
    u_bc: jnp.ndarray,
    x_ic: jnp.ndarray,
    u_ic: jnp.ndarray,
    pde_residual_fn: Callable,
    alpha: float = 0.01,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Single training step.

    Args:
        state: Current training state
        (other args): Training data

    Returns:
        new_state: Updated training state
        loss_dict: Loss components
    """
    new_params, new_opt_state, loss_dict = train_step_jitted(
        state.params,
        state.apply_fn,
        state.opt_state,
        state.tx,
        x_pde,
        t_pde,
        x_bc,
        t_bc,
        u_bc,
        x_ic,
        u_ic,
        alpha,
        state.pde_weight,
        state.bc_weight,
        state.ic_weight,
    )

    new_state = state.replace(params=new_params, opt_state=new_opt_state)
    return new_state, loss_dict


def train_pinn(
    state: TrainState,
    x_pde: jnp.ndarray,
    t_pde: jnp.ndarray,
    x_bc: jnp.ndarray,
    t_bc: jnp.ndarray,
    u_bc: jnp.ndarray,
    x_ic: jnp.ndarray,
    u_ic: jnp.ndarray,
    pde_residual_fn: Callable,
    alpha: float = 0.01,
    num_epochs: int = 10000,
    print_every: int = 1000,
    curriculum_schedule: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[TrainState, Dict[str, list]]:
    """Train PINN with optional curriculum learning.

    Args:
        state: Initial training state
        (data args): Training data
        pde_residual_fn: PDE residual function
        alpha: Thermal diffusivity
        num_epochs: Number of training epochs
        print_every: Print frequency
        curriculum_schedule: Dict mapping epoch -> loss weights
            Example: {0: {"ic": 10, "bc": 1, "pde": 0.1},
                     1000: {"ic": 1, "bc": 1, "pde": 1}}

    Returns:
        trained_state: Final training state
        history: Training history
    """
    history = {"total": [], "pde": [], "bc": [], "ic": []}

    for epoch in tqdm(range(num_epochs), desc="Training PINN"):
        # Update curriculum weights if scheduled
        if curriculum_schedule and epoch in curriculum_schedule:
            weights = curriculum_schedule[epoch]
            state = state.replace(
                pde_weight=weights.get("pde", state.pde_weight),
                bc_weight=weights.get("bc", state.bc_weight),
                ic_weight=weights.get("ic", state.ic_weight),
            )

        # Training step
        state, loss_dict = train_step(
            state, x_pde, t_pde, x_bc, t_bc, u_bc, x_ic, u_ic, pde_residual_fn, alpha
        )

        # Log history
        for key in history:
            history[key].append(float(loss_dict[key]))

        # Print progress
        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Loss: {loss_dict['total']:.6f} | "
                f"PDE: {loss_dict['pde']:.6f} | "
                f"BC: {loss_dict['bc']:.6f} | "
                f"IC: {loss_dict['ic']:.6f}"
            )

    return state, history
