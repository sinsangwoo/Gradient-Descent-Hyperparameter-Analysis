"""Multi-fidelity PINN solver.

Combines low-fidelity (fast, coarse) and high-fidelity (accurate, fine)
simulations for optimal speed-accuracy tradeoff.
"""

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from phio.solvers.pinn_trainer import create_train_state, train_pinn


class MultiFidelitySolver:
    """Multi-fidelity PINN solver.

    Training pipeline:
    1. Low-fidelity: Coarse grid, fewer epochs (fast)
    2. High-fidelity: Fine grid, more epochs (accurate)

    Args:
        model: Flax neural network model
        alpha: Thermal diffusivity for heat equation
        pde_residual_fn: PDE residual function
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.01,
        pde_residual_fn: Callable = None,
    ):
        self.model = model
        self.alpha = alpha
        self.pde_residual_fn = pde_residual_fn

    def generate_collocation_points(
        self,
        rng: jax.random.PRNGKey,
        n_pde: int,
        n_bc: int,
        n_ic: int,
    ) -> Dict[str, jnp.ndarray]:
        """Generate collocation points for training.

        Args:
            rng: Random key
            n_pde: Number of PDE collocation points
            n_bc: Number of boundary points
            n_ic: Number of initial condition points

        Returns:
            Dictionary with training data
        """
        rng, key_pde, key_bc, key_ic = jax.random.split(rng, 4)

        # PDE points (interior domain)
        x_pde = jax.random.uniform(key_pde, (n_pde,))
        t_pde = jax.random.uniform(key_pde, (n_pde,))

        # Boundary conditions (x=0 and x=1)
        t_bc = jax.random.uniform(key_bc, (n_bc,))
        x_bc_left = jnp.zeros(n_bc // 2)
        x_bc_right = jnp.ones(n_bc // 2)
        x_bc = jnp.concatenate([x_bc_left, x_bc_right])
        t_bc_full = jnp.concatenate([t_bc[: n_bc // 2], t_bc[n_bc // 2 :]])
        u_bc = jnp.zeros(n_bc)

        # Initial condition (t=0)
        x_ic = jax.random.uniform(key_ic, (n_ic,))
        u_ic = jnp.sin(jnp.pi * x_ic)

        return {
            "x_pde": x_pde,
            "t_pde": t_pde,
            "x_bc": x_bc,
            "t_bc": t_bc_full,
            "u_bc": u_bc,
            "x_ic": x_ic,
            "u_ic": u_ic,
        }

    def train_low_fidelity(
        self,
        rng: jax.random.PRNGKey,
        n_pde: int = 100,
        n_bc: int = 20,
        n_ic: int = 20,
        num_epochs: int = 500,
        learning_rate: float = 1e-3,
    ) -> Tuple[any, Dict]:
        """Train low-fidelity model (coarse grid, fast).

        Args:
            rng: Random key
            n_pde: Number of PDE points (small for speed)
            n_bc: Number of boundary points
            n_ic: Number of initial condition points
            num_epochs: Training epochs (few for speed)
            learning_rate: Learning rate

        Returns:
            state: Trained model state
            history: Training history
        """
        print("Training low-fidelity model...")
        print(f"  Grid: {n_pde} PDE points, {n_bc} BC, {n_ic} IC")
        print(f"  Epochs: {num_epochs}")

        data = self.generate_collocation_points(rng, n_pde, n_bc, n_ic)

        state = create_train_state(rng, self.model, learning_rate=learning_rate)
        state, history = train_pinn(
            state,
            data["x_pde"],
            data["t_pde"],
            data["x_bc"],
            data["t_bc"],
            data["u_bc"],
            data["x_ic"],
            data["u_ic"],
            pde_residual_fn=self.pde_residual_fn,
            alpha=self.alpha,
            num_epochs=num_epochs,
            print_every=num_epochs // 5,
        )

        print(f"Low-fidelity complete. Final loss: {history['total'][-1]:.6e}")
        return state, history

    def train_high_fidelity(
        self,
        rng: jax.random.PRNGKey,
        initial_state: any,
        n_pde: int = 1000,
        n_bc: int = 100,
        n_ic: int = 100,
        num_epochs: int = 2000,
        learning_rate: float = 1e-4,
    ) -> Tuple[any, Dict]:
        """Train high-fidelity model (fine grid, accurate).

        Args:
            rng: Random key
            initial_state: Low-fidelity state to start from
            n_pde: Number of PDE points (large for accuracy)
            n_bc: Number of boundary points
            n_ic: Number of initial condition points
            num_epochs: Training epochs (many for accuracy)
            learning_rate: Learning rate (smaller for refinement)

        Returns:
            state: Trained model state
            history: Training history
        """
        print("\nTraining high-fidelity model...")
        print(f"  Grid: {n_pde} PDE points, {n_bc} BC, {n_ic} IC")
        print(f"  Epochs: {num_epochs}")

        data = self.generate_collocation_points(rng, n_pde, n_bc, n_ic)

        # Start from low-fidelity parameters
        state = initial_state.replace(
            tx=initial_state.tx.__class__(learning_rate),
            opt_state=initial_state.tx.__class__(learning_rate).init(
                initial_state.params
            ),
        )

        state, history = train_pinn(
            state,
            data["x_pde"],
            data["t_pde"],
            data["x_bc"],
            data["t_bc"],
            data["u_bc"],
            data["x_ic"],
            data["u_ic"],
            pde_residual_fn=self.pde_residual_fn,
            alpha=self.alpha,
            num_epochs=num_epochs,
            print_every=num_epochs // 5,
        )

        print(f"High-fidelity complete. Final loss: {history['total'][-1]:.6e}")
        return state, history

    def multifidelity_pipeline(
        self,
        rng: jax.random.PRNGKey,
        initial_condition: Callable,
        analytical_solution: Callable = None,
    ) -> Dict:
        """Full multi-fidelity training pipeline.

        Args:
            rng: Random key
            initial_condition: Initial condition function
            analytical_solution: Optional analytical solution for error

        Returns:
            results: Dictionary with states, histories, metrics
        """
        print("="*60)
        print("MULTI-FIDELITY TRAINING PIPELINE")
        print("="*60)

        # Low-fidelity phase
        rng, key_low = jax.random.split(rng)
        state_low, history_low = self.train_low_fidelity(key_low)

        # High-fidelity phase
        rng, key_high = jax.random.split(rng)
        state_high, history_high = self.train_high_fidelity(
            key_high, state_low
        )

        # Compute metrics if analytical solution provided
        results = {
            "state_low": state_low,
            "state_high": state_high,
            "history_low": history_low,
            "history_high": history_high,
        }

        if analytical_solution is not None:
            # Test points
            x_test = jnp.linspace(0, 1, 100)
            t_test = jnp.ones(100) * 0.5
            u_true = analytical_solution(x_test, t_test)

            # Low-fidelity error
            u_pred_low = state_low.apply_fn(
                state_low.params, x_test[:, None], t_test[:, None]
            ).squeeze()
            error_low = jnp.mean(jnp.abs(u_pred_low - u_true))

            # High-fidelity error
            u_pred_high = state_high.apply_fn(
                state_high.params, x_test[:, None], t_test[:, None]
            ).squeeze()
            error_high = jnp.mean(jnp.abs(u_pred_high - u_true))

            error_reduction = (error_low - error_high) / error_low * 100

            results.update({
                "error_low": float(error_low),
                "error_high": float(error_high),
                "error_reduction_percent": float(error_reduction),
            })

            print("\n" + "="*60)
            print("RESULTS")
            print("="*60)
            print(f"Low-fidelity error:  {error_low:.6e}")
            print(f"High-fidelity error: {error_high:.6e}")
            print(f"Error reduction:     {error_reduction:.2f}%")

            # Cost function: accuracy per computational cost
            # Assume high-fidelity is 10x more expensive
            cost_low = 500
            cost_high = 2000 * 10
            total_cost = cost_low + cost_high

            # Cost function: 1 / (error * cost)
            cost_function = 1.0 / (error_high * total_cost)
            results["cost_function"] = float(cost_function)

            print(f"\nCost function (1/error*cost): {cost_function:.6e}")
            print("="*60)

        return results
