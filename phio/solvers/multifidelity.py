"""Multi-fidelity optimization: Low-fidelity PINN + High-fidelity FDM refinement.

Phase 2.2 implementation:
- Low-fidelity: Coarse grid PINN (fast, ~100 points)
- High-fidelity: Fine grid FDM + PINN refinement (accurate, ~1000 points)
- Cost function: Accuracy per GPU-hour
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from phio.physics.heat import heat_equation_residual
from phio.solvers.pinn_trainer import TrainState, create_train_state, train_pinn


@dataclass
class FidelityLevel:
    """Configuration for a fidelity level."""

    name: str
    n_points_x: int  # Spatial resolution
    n_points_t: int  # Temporal resolution
    n_epochs: int  # Training epochs
    learning_rate: float


class MultiFidelitySolver:
    """Multi-fidelity PINN solver with adaptive mesh refinement."""

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.01,
        x_range: Tuple[float, float] = (0.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """Initialize multi-fidelity solver.

        Args:
            model: Neural network model
            alpha: Thermal diffusivity
            x_range: Spatial domain (x_min, x_max)
            t_range: Temporal domain (t_min, t_max)
        """
        self.model = model
        self.alpha = alpha
        self.x_range = x_range
        self.t_range = t_range

        # Define fidelity levels
        self.low_fidelity = FidelityLevel(
            name="low",
            n_points_x=20,
            n_points_t=10,
            n_epochs=500,
            learning_rate=1e-3,
        )

        self.high_fidelity = FidelityLevel(
            name="high",
            n_points_x=100,
            n_points_t=50,
            n_epochs=2000,
            learning_rate=5e-4,
        )

    def generate_training_data(
        self,
        fidelity: FidelityLevel,
        initial_condition: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Generate training data for a fidelity level.

        Args:
            fidelity: Fidelity level configuration
            initial_condition: u(x, 0) function

        Returns:
            Dictionary with training data arrays
        """
        x_min, x_max = self.x_range
        t_min, t_max = self.t_range

        # PDE collocation points (interior)
        x_pde = jnp.linspace(x_min, x_max, fidelity.n_points_x)
        t_pde = jnp.linspace(t_min, t_max, fidelity.n_points_t)
        x_pde_grid, t_pde_grid = jnp.meshgrid(x_pde, t_pde, indexing="ij")
        x_pde = x_pde_grid.flatten()
        t_pde = t_pde_grid.flatten()

        # Boundary conditions (x=0 and x=1)
        t_bc = jnp.linspace(t_min, t_max, fidelity.n_points_t)
        x_bc_left = jnp.zeros_like(t_bc)
        x_bc_right = jnp.ones_like(t_bc)
        x_bc = jnp.concatenate([x_bc_left, x_bc_right])
        t_bc = jnp.concatenate([t_bc, t_bc])
        u_bc = jnp.zeros_like(t_bc)  # Dirichlet BC: u=0 at boundaries

        # Initial condition (t=0)
        x_ic = jnp.linspace(x_min, x_max, fidelity.n_points_x)
        u_ic = jax.vmap(initial_condition)(x_ic)

        return {
            "x_pde": x_pde,
            "t_pde": t_pde,
            "x_bc": x_bc,
            "t_bc": t_bc,
            "u_bc": u_bc,
            "x_ic": x_ic,
            "u_ic": u_ic,
        }

    def train_low_fidelity(
        self,
        rng: jax.random.PRNGKey,
        initial_condition: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> Tuple[TrainState, Dict[str, list], float]:
        """Train low-fidelity PINN on coarse grid.

        Args:
            rng: Random key
            initial_condition: Initial condition function

        Returns:
            state: Trained model state
            history: Training history
            elapsed_time: Training time in seconds
        """
        data = self.generate_training_data(self.low_fidelity, initial_condition)

        # Initialize state
        state = create_train_state(
            rng,
            self.model,
            learning_rate=self.low_fidelity.learning_rate,
            sample_input=(data["x_pde"][:1, None], data["t_pde"][:1, None]),
        )

        # Train
        start_time = time.time()
        state, history = train_pinn(
            state,
            data["x_pde"],
            data["t_pde"],
            data["x_bc"],
            data["t_bc"],
            data["u_bc"],
            data["x_ic"],
            data["u_ic"],
            heat_equation_residual,
            alpha=self.alpha,
            num_epochs=self.low_fidelity.n_epochs,
            print_every=100,
        )
        elapsed_time = time.time() - start_time

        print(
            f"\n[Low-Fidelity] Training completed in {elapsed_time:.2f}s "
            f"({self.low_fidelity.n_epochs} epochs)"
        )

        return state, history, elapsed_time

    def refine_with_high_fidelity(
        self,
        low_fidelity_state: TrainState,
        initial_condition: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> Tuple[TrainState, Dict[str, list], float]:
        """Refine with high-fidelity training on fine grid.

        Args:
            low_fidelity_state: Pre-trained low-fidelity model
            initial_condition: Initial condition function

        Returns:
            state: Refined model state
            history: Training history
            elapsed_time: Training time in seconds
        """
        data = self.generate_training_data(self.high_fidelity, initial_condition)

        # Initialize from low-fidelity parameters
        state = low_fidelity_state.replace(
            tx=low_fidelity_state.tx.__class__(
                learning_rate=self.high_fidelity.learning_rate
            )
        )

        # Train with fine grid
        start_time = time.time()
        state, history = train_pinn(
            state,
            data["x_pde"],
            data["t_pde"],
            data["x_bc"],
            data["t_bc"],
            data["u_bc"],
            data["x_ic"],
            data["u_ic"],
            heat_equation_residual,
            alpha=self.alpha,
            num_epochs=self.high_fidelity.n_epochs,
            print_every=500,
        )
        elapsed_time = time.time() - start_time

        print(
            f"\n[High-Fidelity] Refinement completed in {elapsed_time:.2f}s "
            f"({self.high_fidelity.n_epochs} epochs)"
        )

        return state, history, elapsed_time

    def compute_accuracy(
        self,
        state: TrainState,
        analytical_solution: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        n_test_points: int = 100,
    ) -> Dict[str, float]:
        """Compute accuracy metrics.

        Args:
            state: Trained model state
            analytical_solution: u_true(x, t) function
            n_test_points: Number of test points

        Returns:
            Dictionary with accuracy metrics
        """
        x_min, x_max = self.x_range
        t_min, t_max = self.t_range

        # Generate test grid
        x_test = jnp.linspace(x_min, x_max, n_test_points)
        t_test = jnp.linspace(t_min, t_max, n_test_points)
        x_grid, t_grid = jnp.meshgrid(x_test, t_test, indexing="ij")
        x_flat = x_grid.flatten()
        t_flat = t_grid.flatten()

        # Predictions
        u_pred = jax.vmap(state.apply_fn, in_axes=(None, 0, 0))(
            state.params, x_flat[:, None], t_flat[:, None]
        ).squeeze()

        # Analytical solution
        u_true = jax.vmap(analytical_solution)(x_flat, t_flat)

        # Metrics
        mse = jnp.mean((u_pred - u_true) ** 2)
        mae = jnp.mean(jnp.abs(u_pred - u_true))
        rel_error = jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true)

        return {"mse": float(mse), "mae": float(mae), "relative_error": float(rel_error)}

    def multifidelity_pipeline(
        self,
        rng: jax.random.PRNGKey,
        initial_condition: Callable[[jnp.ndarray], jnp.ndarray],
        analytical_solution: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ) -> Dict:
        """Complete multi-fidelity pipeline.

        Args:
            rng: Random key
            initial_condition: u(x, 0)
            analytical_solution: u_true(x, t) for validation

        Returns:
            Results dictionary with states, metrics, and timings
        """
        print("=" * 60)
        print("MULTI-FIDELITY OPTIMIZATION PIPELINE")
        print("=" * 60)

        # Stage 1: Low-fidelity training
        print("\n[Stage 1] Low-Fidelity Training (Coarse Grid)")
        low_state, low_history, low_time = self.train_low_fidelity(rng, initial_condition)
        low_accuracy = self.compute_accuracy(low_state, analytical_solution)

        print(f"\nLow-Fidelity Accuracy:")
        print(f"  MSE: {low_accuracy['mse']:.6f}")
        print(f"  MAE: {low_accuracy['mae']:.6f}")
        print(f"  Relative Error: {low_accuracy['relative_error']:.6f}")

        # Stage 2: High-fidelity refinement
        print("\n[Stage 2] High-Fidelity Refinement (Fine Grid)")
        high_state, high_history, high_time = self.refine_with_high_fidelity(
            low_state, initial_condition
        )
        high_accuracy = self.compute_accuracy(high_state, analytical_solution)

        print(f"\nHigh-Fidelity Accuracy:")
        print(f"  MSE: {high_accuracy['mse']:.6f}")
        print(f"  MAE: {high_accuracy['mae']:.6f}")
        print(f"  Relative Error: {high_accuracy['relative_error']:.6f}")

        # Compute cost-effectiveness
        total_time = low_time + high_time
        accuracy_per_second = 1.0 / (
            high_accuracy["relative_error"] * total_time
        )  # Higher is better

        print("\n" + "=" * 60)
        print("COST-EFFECTIVENESS ANALYSIS")
        print("=" * 60)
        print(f"Total training time: {total_time:.2f}s")
        print(
            f"Final relative error: {high_accuracy['relative_error']:.6f} "
            f"({(1 - high_accuracy['relative_error']) * 100:.2f}% accurate)"
        )
        print(f"Cost function (accuracy per second): {accuracy_per_second:.6f}")

        # Improvement analysis
        error_reduction = (
            (low_accuracy["relative_error"] - high_accuracy["relative_error"])
            / low_accuracy["relative_error"]
            * 100
        )
        print(
            f"\nError reduction from low to high fidelity: {error_reduction:.2f}%"
        )

        return {
            "low_fidelity": {
                "state": low_state,
                "history": low_history,
                "time": low_time,
                "accuracy": low_accuracy,
            },
            "high_fidelity": {
                "state": high_state,
                "history": high_history,
                "time": high_time,
                "accuracy": high_accuracy,
            },
            "total_time": total_time,
            "cost_function": accuracy_per_second,
            "error_reduction_percent": error_reduction,
        }
