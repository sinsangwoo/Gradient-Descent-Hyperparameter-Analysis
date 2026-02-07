"""Inverse problem solver for parameter estimation from experimental data.

Phase 2.2: Given measurement data (temperature, pressure, velocity),
find hidden parameters (thermal conductivity, viscosity, boundary conditions).

Use case: Materials discovery, process optimization
"""

from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import tqdm

from phio.solvers.pinn_trainer import TrainState, create_train_state


class InverseProblemSolver:
    """Solve inverse problems: infer physical parameters from measurements."""

    def __init__(
        self,
        model: nn.Module,
        pde_residual_fn: Callable,
        x_range: Tuple[float, float] = (0.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """Initialize inverse problem solver.

        Args:
            model: Neural network model
            pde_residual_fn: PDE residual function (must accept parameters)
            x_range: Spatial domain
            t_range: Temporal domain
        """
        self.model = model
        self.pde_residual_fn = pde_residual_fn
        self.x_range = x_range
        self.t_range = t_range

    def compute_inverse_loss(
        self,
        state: TrainState,
        physical_params: Dict[str, jnp.ndarray],
        x_pde: jnp.ndarray,
        t_pde: jnp.ndarray,
        x_bc: jnp.ndarray,
        t_bc: jnp.ndarray,
        u_bc: jnp.ndarray,
        x_ic: jnp.ndarray,
        u_ic: jnp.ndarray,
        x_meas: jnp.ndarray,
        t_meas: jnp.ndarray,
        u_meas: jnp.ndarray,
        data_weight: float = 10.0,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute inverse problem loss.

        Loss = L_data + L_pde + L_bc + L_ic
        where L_data = ||u_pred(x_meas, t_meas) - u_meas||^2

        Args:
            state: Model state
            physical_params: Dictionary of physical parameters to estimate
            (data args): Training and measurement data
            data_weight: Weight for measurement data loss

        Returns:
            total_loss: Weighted loss
            loss_dict: Individual loss components
        """
        # Extract parameters
        params = state.params
        alpha = physical_params["alpha"]  # Thermal diffusivity to be estimated

        # 1. Data fidelity loss (match measurements)
        u_pred_meas = jax.vmap(state.apply_fn, in_axes=(None, 0, 0))(
            params, x_meas[:, None], t_meas[:, None]
        ).squeeze()
        loss_data = jnp.mean((u_pred_meas - u_meas) ** 2)

        # 2. PDE residual loss
        def predict_single(params_inner, x, t):
            return state.apply_fn(params_inner, x[None, :], t[None, :])[0]

        residual = jax.vmap(
            lambda x, t: self.pde_residual_fn(predict_single, params, x, t, alpha=alpha)
        )(x_pde, t_pde)
        loss_pde = jnp.mean(residual**2)

        # 3. Boundary condition loss
        u_bc_pred = jax.vmap(state.apply_fn, in_axes=(None, 0, 0))(
            params, x_bc[:, None], t_bc[:, None]
        ).squeeze()
        loss_bc = jnp.mean((u_bc_pred - u_bc) ** 2)

        # 4. Initial condition loss
        t_ic_full = jnp.zeros_like(x_ic)
        u_ic_pred = jax.vmap(state.apply_fn, in_axes=(None, 0, 0))(
            params, x_ic[:, None], t_ic_full[:, None]
        ).squeeze()
        loss_ic = jnp.mean((u_ic_pred - u_ic) ** 2)

        # Total loss
        total_loss = data_weight * loss_data + loss_pde + loss_bc + loss_ic

        loss_dict = {
            "total": total_loss,
            "data": loss_data,
            "pde": loss_pde,
            "bc": loss_bc,
            "ic": loss_ic,
        }

        return total_loss, loss_dict

    def inverse_train_step(
        self,
        state: TrainState,
        physical_params: Dict[str, jnp.ndarray],
        data: Dict[str, jnp.ndarray],
        data_weight: float = 10.0,
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
        """Single training step for inverse problem.

        Jointly optimizes neural network parameters AND physical parameters.

        Args:
            state: Model state
            physical_params: Dictionary of physical parameters
            data: Training data
            data_weight: Weight for measurement data

        Returns:
            new_state: Updated model state
            new_physical_params: Updated physical parameters
            loss_dict: Loss components
        """

        def loss_fn(params_and_physics):
            nn_params, physics = params_and_physics
            temp_state = state.replace(params=nn_params)
            return self.compute_inverse_loss(
                temp_state,
                physics,
                data["x_pde"],
                data["t_pde"],
                data["x_bc"],
                data["t_bc"],
                data["u_bc"],
                data["x_ic"],
                data["u_ic"],
                data["x_meas"],
                data["t_meas"],
                data["u_meas"],
                data_weight=data_weight,
            )

        # Compute gradients w.r.t. both NN params and physical params
        (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            (state.params, physical_params)
        )

        # Update NN parameters
        grad_nn, grad_physics = grads
        new_state = state.apply_gradients(grads=grad_nn)

        # Update physical parameters
        new_physical_params = {
            key: val - 1e-4 * grad_physics[key]  # Simple SGD for physical params
            for key, val in physical_params.items()
        }

        return new_state, new_physical_params, loss_dict

    def solve_inverse_problem(
        self,
        rng: jax.random.PRNGKey,
        x_measurements: jnp.ndarray,
        t_measurements: jnp.ndarray,
        u_measurements: jnp.ndarray,
        initial_condition: Callable[[jnp.ndarray], jnp.ndarray],
        initial_param_guess: Dict[str, float],
        n_epochs: int = 2000,
        n_collocation_points: int = 100,
        data_weight: float = 10.0,
        print_every: int = 200,
    ) -> Tuple[TrainState, Dict[str, float], Dict[str, List[float]]]:
        """Solve inverse problem: estimate parameters from measurements.

        Args:
            rng: Random key
            x_measurements: Measurement locations (space)
            t_measurements: Measurement times
            u_measurements: Measured values
            initial_condition: u(x, 0) function
            initial_param_guess: Initial guess for physical parameters
            n_epochs: Number of training epochs
            n_collocation_points: Number of PDE collocation points
            data_weight: Weight for measurement data loss
            print_every: Print frequency

        Returns:
            state: Trained model state
            estimated_params: Estimated physical parameters
            history: Training history
        """
        x_min, x_max = self.x_range
        t_min, t_max = self.t_range

        # Generate training data
        x_pde = jax.random.uniform(
            rng, shape=(n_collocation_points,), minval=x_min, maxval=x_max
        )
        t_pde = jax.random.uniform(
            rng, shape=(n_collocation_points,), minval=t_min, maxval=t_max
        )

        # Boundary conditions
        n_bc = 20
        t_bc = jnp.linspace(t_min, t_max, n_bc)
        x_bc = jnp.concatenate([jnp.zeros(n_bc), jnp.ones(n_bc)])
        t_bc = jnp.concatenate([t_bc, t_bc])
        u_bc = jnp.zeros_like(t_bc)

        # Initial condition
        x_ic = jnp.linspace(x_min, x_max, 50)
        u_ic = jax.vmap(initial_condition)(x_ic)

        data = {
            "x_pde": x_pde,
            "t_pde": t_pde,
            "x_bc": x_bc,
            "t_bc": t_bc,
            "u_bc": u_bc,
            "x_ic": x_ic,
            "u_ic": u_ic,
            "x_meas": x_measurements,
            "t_meas": t_measurements,
            "u_meas": u_measurements,
        }

        # Initialize state and parameters
        state = create_train_state(
            rng,
            self.model,
            learning_rate=1e-3,
            sample_input=(x_pde[:1, None], t_pde[:1, None]),
        )
        physical_params = {
            key: jnp.array(val) for key, val in initial_param_guess.items()
        }

        # Training history
        history = {
            "total": [],
            "data": [],
            "pde": [],
            "bc": [],
            "ic": [],
            "alpha": [],
        }

        print("=" * 60)
        print("INVERSE PROBLEM: PARAMETER ESTIMATION FROM MEASUREMENTS")
        print("=" * 60)
        print(f"Initial parameter guess: {initial_param_guess}")
        print(f"Number of measurements: {len(u_measurements)}")
        print(f"Training for {n_epochs} epochs...\n")

        # Training loop
        for epoch in tqdm(range(n_epochs), desc="Training Inverse Problem"):
            state, physical_params, loss_dict = self.inverse_train_step(
                state, physical_params, data, data_weight=data_weight
            )

            # Log history
            for key in ["total", "data", "pde", "bc", "ic"]:
                history[key].append(float(loss_dict[key]))
            history["alpha"].append(float(physical_params["alpha"]))

            # Print progress
            if (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Loss: {loss_dict['total']:.6f} | "
                    f"Data: {loss_dict['data']:.6f} | "
                    f"PDE: {loss_dict['pde']:.6f} | "
                    f"α (estimated): {physical_params['alpha']:.6f}"
                )

        # Final results
        estimated_params = {key: float(val) for key, val in physical_params.items()}

        print("\n" + "=" * 60)
        print("ESTIMATION RESULTS")
        print("=" * 60)
        print(f"Initial guess: {initial_param_guess}")
        print(f"Final estimate: {estimated_params}")
        print("=" * 60)

        return state, estimated_params, history
