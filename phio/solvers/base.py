"""Base PINN solver implementation using JAX and Flax."""

from typing import Dict, List, Tuple, Callable, Optional
import time

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import flax.linen as nn
import optax

from phio.core import PDE, BoundaryCondition, InitialCondition
from phio.networks import MLP
from phio.losses import pinn_loss
from phio.utils import logger


class PINNSolver:
    """Physics-Informed Neural Network solver.

    Solves PDEs by training a neural network to satisfy:
    1. PDE residual = 0 in the domain
    2. Boundary conditions at domain boundaries
    3. Initial conditions at t=0 (for time-dependent problems)

    Args:
        pde: PDE instance to solve
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name ('tanh', 'relu', 'gelu')
        optimizer: Optimizer name ('adam', 'sgd', 'adamw')
        learning_rate: Initial learning rate
        seed: Random seed for reproducibility

    Example:
        >>> from phio.physics import HeatEquation1D
        >>> from phio.solvers import PINNSolver
        >>> from phio.core import DirichletBC, InitialCondition
        >>>
        >>> # Define problem
        >>> pde = HeatEquation1D(domain=(0, 1), diffusion_coeff=0.01)
        >>> bc_left = DirichletBC('left', lambda t: 0.0)
        >>> bc_right = DirichletBC('right', lambda t: 0.0)
        >>> ic = InitialCondition(lambda x: jnp.sin(jnp.pi * x))
        >>>
        >>> # Create solver
        >>> solver = PINNSolver(pde, hidden_dims=[64, 64, 64])
        >>> solver.set_boundary_conditions([bc_left, bc_right])
        >>> solver.set_initial_condition(ic)
        >>>
        >>> # Train
        >>> results = solver.train(num_epochs=10000, n_collocation=1000)
    """

    def __init__(
        self,
        pde: PDE,
        hidden_dims: List[int] = [64, 64, 64],
        activation: str = "tanh",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        seed: int = 42,
    ):
        self.pde = pde
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.seed = seed

        # Initialize network
        self.network = MLP(hidden_dims=hidden_dims, output_dim=1, activation=activation)
        self.rng = random.PRNGKey(seed)

        # Initialize optimizer
        if optimizer == "adam":
            self.optimizer = optax.adam(learning_rate)
        elif optimizer == "sgd":
            self.optimizer = optax.sgd(learning_rate)
        elif optimizer == "adamw":
            self.optimizer = optax.adamw(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Boundary and initial conditions
        self.boundary_conditions: List[BoundaryCondition] = []
        self.initial_condition: Optional[InitialCondition] = None

        # Training state
        self.params = None
        self.opt_state = None
        self.loss_history = []

    def set_boundary_conditions(self, bcs: List[BoundaryCondition]):
        """Set boundary conditions for the problem."""
        self.boundary_conditions = bcs

    def set_initial_condition(self, ic: InitialCondition):
        """Set initial condition for time-dependent problems."""
        self.initial_condition = ic

    def initialize_params(self, sample_input: jnp.ndarray):
        """Initialize network parameters.

        Args:
            sample_input: Sample input for parameter initialization (shape: [batch, 2])
        """
        self.rng, init_rng = random.split(self.rng)
        self.params = self.network.init(init_rng, sample_input)
        self.opt_state = self.optimizer.init(self.params)
        logger.info(f"Initialized network with {self._count_params()} parameters")

    def _count_params(self) -> int:
        """Count total number of trainable parameters."""
        return sum(x.size for x in jax.tree_util.tree_leaves(self.params))

    def train(
        self,
        num_epochs: int = 10000,
        n_collocation: int = 1000,
        n_boundary: int = 100,
        n_initial: int = 100,
        log_frequency: int = 100,
        loss_weights: Dict[str, float] = None,
    ) -> Dict[str, any]:
        """Train the PINN.

        Args:
            num_epochs: Number of training iterations
            n_collocation: Number of collocation points for PDE residual
            n_boundary: Number of boundary points per boundary
            n_initial: Number of initial condition points
            log_frequency: Print loss every N epochs
            loss_weights: Weights for loss components {'pde': 1.0, 'bc': 1.0, 'ic': 1.0}

        Returns:
            Dictionary with training history and final metrics
        """
        if loss_weights is None:
            loss_weights = {"pde": 1.0, "bc": 1.0, "ic": 1.0}

        # Generate sample input for initialization
        self.rng, sample_rng = random.split(self.rng)
        sample_input = random.uniform(sample_rng, (10, 2))  # [x, t]
        self.initialize_params(sample_input)

        # Training loop
        start_time = time.time()
        self.loss_history = []

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Collocation points: {n_collocation}, Boundary: {n_boundary}, Initial: {n_initial}")

        for epoch in range(num_epochs):
            # Generate training points
            self.rng, *rngs = random.split(self.rng, 4)
            x_colloc, t_colloc = self._sample_collocation_points(rngs[0], n_collocation)
            x_bc, t_bc = self._sample_boundary_points(rngs[1], n_boundary)
            x_ic, t_ic = self._sample_initial_points(rngs[2], n_initial)

            # Training step
            loss_value, self.params, self.opt_state = self._train_step(
                self.params,
                self.opt_state,
                x_colloc,
                t_colloc,
                x_bc,
                t_bc,
                x_ic,
                t_ic,
                loss_weights,
            )

            self.loss_history.append(float(loss_value))

            if (epoch + 1) % log_frequency == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss_value:.6e}")

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f}s")

        return {
            "loss_history": self.loss_history,
            "final_loss": self.loss_history[-1],
            "training_time": training_time,
            "num_params": self._count_params(),
        }

    def _sample_collocation_points(
        self, rng: jax.random.PRNGKey, n_points: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample random collocation points in the domain."""
        rng_x, rng_t = random.split(rng)
        x_min, x_max = self.pde.domain
        t_min, t_max = self.pde.time_domain

        x = random.uniform(rng_x, (n_points,), minval=x_min, maxval=x_max)
        t = random.uniform(rng_t, (n_points,), minval=t_min, maxval=t_max)

        return x, t

    def _sample_boundary_points(
        self, rng: jax.random.PRNGKey, n_points: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample points at domain boundaries."""
        x_min, x_max = self.pde.domain
        t_min, t_max = self.pde.time_domain

        t = random.uniform(rng, (n_points * 2,), minval=t_min, maxval=t_max)
        x_left = jnp.full((n_points,), x_min)
        x_right = jnp.full((n_points,), x_max)
        x = jnp.concatenate([x_left, x_right])

        return x, t

    def _sample_initial_points(
        self, rng: jax.random.PRNGKey, n_points: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample points at initial time t=0."""
        x_min, x_max = self.pde.domain
        t_min, _ = self.pde.time_domain

        x = random.uniform(rng, (n_points,), minval=x_min, maxval=x_max)
        t = jnp.full((n_points,), t_min)

        return x, t

    @staticmethod
    @jit
    def _train_step(
        params,
        opt_state,
        x_colloc,
        t_colloc,
        x_bc,
        t_bc,
        x_ic,
        t_ic,
        loss_weights,
    ):
        """Single training step (JIT-compiled)."""
        # This is a placeholder - actual implementation will use pinn_loss from losses module
        # For now, return dummy values
        loss_value = 0.0
        return loss_value, params, opt_state

    def predict(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Predict solution at given points.

        Args:
            x: Spatial coordinates
            t: Temporal coordinates

        Returns:
            Predicted solution values
        """
        if self.params is None:
            raise RuntimeError("Model not trained. Call train() first.")

        inputs = jnp.stack([x, t], axis=-1)
        return self.network.apply(self.params, inputs).squeeze()

    def compute_error(self, x: jnp.ndarray, t: jnp.ndarray) -> float:
        """Compute L2 error against exact solution.

        Args:
            x: Test points spatial coordinates
            t: Test points temporal coordinates

        Returns:
            L2 relative error
        """
        u_pred = self.predict(x, t)
        u_exact = self.pde.exact_solution(x, t)

        error = jnp.linalg.norm(u_pred - u_exact) / jnp.linalg.norm(u_exact)
        return float(error)
