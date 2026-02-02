"""Initial condition class for time-dependent PDEs."""

from typing import Callable
import jax.numpy as jnp


class InitialCondition:
    """Initial condition for time-dependent problems.

    Specifies u(x, t=0) = f(x).

    Example:
        >>> # Gaussian initial temperature distribution
        >>> ic = InitialCondition(
        ...     value_fn=lambda x: jnp.exp(-10 * (x - 0.5)**2)
        ... )
    """

    def __init__(self, value_fn: Callable[[jnp.ndarray], jnp.ndarray]):
        """Initialize initial condition.

        Args:
            value_fn: Function that returns initial value given spatial coordinate x
        """
        self.value_fn = value_fn

    def apply(self, u: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Compute initial condition residual.

        Args:
            u: Predicted solution at t=0
            x: Spatial coordinates

        Returns:
            Residual: u(x, 0) - f(x)
        """
        target_value = self.value_fn(x)
        return u - target_value

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate initial condition.

        Args:
            x: Spatial coordinates

        Returns:
            Initial values at x
        """
        return self.value_fn(x)
