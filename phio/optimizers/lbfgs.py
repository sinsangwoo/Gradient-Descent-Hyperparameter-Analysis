"""L-BFGS optimizer for PINN refinement.

L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a
quasi-Newton method that approximates the Hessian using gradient history.

Use case: After Adam pre-training, switch to L-BFGS for final refinement.
L-BFGS excels at finding precise local minima.
"""

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from jax.scipy.optimize import minimize


class LBFGSOptimizer:
    """L-BFGS optimizer wrapper for JAX.

    Two-stage training strategy:
    1. Stage 1 (epochs 0-N): Adam for global exploration
    2. Stage 2 (epochs N+): L-BFGS for local refinement

    This often achieves 10-100x better final accuracy.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-8,
        history_size: int = 50,
    ):
        """Initialize L-BFGS optimizer.

        Args:
            max_iter: Maximum L-BFGS iterations
            tol: Convergence tolerance (gradient norm)
            history_size: Number of gradient/position pairs to store
        """
        self.max_iter = max_iter
        self.tol = tol
        self.history_size = history_size

    def optimize(
        self,
        loss_fn: Callable,
        params_init: dict,
        verbose: bool = True,
    ) -> Tuple[dict, Dict[str, list]]:
        """Run L-BFGS optimization.

        Args:
            loss_fn: Function params -> scalar loss
            params_init: Initial parameters (pytree)
            verbose: Print optimization progress

        Returns:
            optimized_params: Final parameters
            history: Dictionary with loss history
        """
        # Flatten parameters for scipy.optimize interface
        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params_init)

        # Define loss and gradient for flattened parameters
        def flat_loss_fn(flat_p):
            params = unravel_fn(flat_p)
            return loss_fn(params)

        # Track optimization history
        history = {"loss": [], "grad_norm": []}

        def callback(params_flat):
            """Callback for tracking progress."""
            loss = flat_loss_fn(params_flat)
            grad = jax.grad(flat_loss_fn)(params_flat)
            grad_norm = jnp.linalg.norm(grad)

            history["loss"].append(float(loss))
            history["grad_norm"].append(float(grad_norm))

            if verbose and len(history["loss"]) % 100 == 0:
                print(
                    f"L-BFGS iter {len(history['loss'])}: "
                    f"Loss = {loss:.6e}, ||grad|| = {grad_norm:.6e}"
                )

        # Run L-BFGS
        result = minimize(
            fun=flat_loss_fn,
            x0=flat_params,
            method="L-BFGS-B",
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
                "gtol": self.tol,
            },
            callback=callback,
        )

        # Convert back to pytree
        optimized_params = unravel_fn(result.x)

        if verbose:
            print(f"\nL-BFGS converged: {result.success}")
            print(f"Final loss: {result.fun:.6e}")
            print(f"Iterations: {result.nit}")

        return optimized_params, history


def two_stage_training(
    params_init: dict,
    loss_fn: Callable,
    adam_epochs: int = 5000,
    lbfgs_iters: int = 1000,
    learning_rate: float = 1e-3,
    verbose: bool = True,
) -> Tuple[dict, Dict[str, list]]:
    """Two-stage training: Adam + L-BFGS.

    Args:
        params_init: Initial parameters
        loss_fn: Loss function
        adam_epochs: Number of Adam epochs
        lbfgs_iters: Number of L-BFGS iterations
        learning_rate: Adam learning rate
        verbose: Print progress

    Returns:
        final_params: Optimized parameters
        history: Combined training history
    """
    # Stage 1: Adam
    if verbose:
        print("=" * 70)
        print("Stage 1: Adam Pretraining")
        print("=" * 70)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params_init)

    params = params_init
    adam_history = {"loss": [], "grad_norm": []}

    @jax.jit
    def adam_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads

    for epoch in range(adam_epochs):
        params, opt_state, loss, grads = adam_step(params, opt_state)

        # Track history
        flat_grads = jax.tree_util.tree_leaves(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in flat_grads))

        adam_history["loss"].append(float(loss))
        adam_history["grad_norm"].append(float(grad_norm))

        if verbose and (epoch + 1) % 1000 == 0:
            print(
                f"Adam epoch {epoch+1}/{adam_epochs}: "
                f"Loss = {loss:.6e}, ||grad|| = {grad_norm:.6e}"
            )

    # Stage 2: L-BFGS
    if verbose:
        print("\n" + "=" * 70)
        print("Stage 2: L-BFGS Refinement")
        print("=" * 70)

    lbfgs = LBFGSOptimizer(max_iter=lbfgs_iters)
    final_params, lbfgs_history = lbfgs.optimize(loss_fn, params, verbose=verbose)

    # Combine histories
    combined_history = {
        "adam_loss": adam_history["loss"],
        "adam_grad_norm": adam_history["grad_norm"],
        "lbfgs_loss": lbfgs_history["loss"],
        "lbfgs_grad_norm": lbfgs_history["grad_norm"],
    }

    return final_params, combined_history
