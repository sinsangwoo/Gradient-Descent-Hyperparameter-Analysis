"""Boundary condition classes for PDEs."""

from abc import ABC, abstractmethod
from typing import Callable, Union
import jax.numpy as jnp


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions."""

    def __init__(self, location: Union[str, float], value_fn: Callable):
        """Initialize boundary condition.

        Args:
            location: Boundary location (e.g., 'left', 'right', or coordinate)
            value_fn: Function that returns boundary value given time t
        """
        self.location = location
        self.value_fn = value_fn

    @abstractmethod
    def apply(self, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute boundary condition residual.

        Args:
            u: Solution values at boundary
            t: Time values

        Returns:
            Residual (should be zero when BC is satisfied)
        """
        pass


class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition: u(x_boundary, t) = g(t).

    Example:
        >>> # u(0, t) = sin(t)
        >>> bc_left = DirichletBC(location='left', value_fn=lambda t: jnp.sin(t))
    """

    def apply(self, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Enforce u = g(t) at boundary.

        Args:
            u: Predicted solution at boundary
            t: Time values

        Returns:
            Residual: u - g(t)
        """
        target_value = self.value_fn(t)
        return u - target_value


class NeumannBC(BoundaryCondition):
    """Neumann boundary condition: du/dx(x_boundary, t) = g(t).

    Example:
        >>> # du/dx(1, t) = 0 (insulated boundary)
        >>> bc_right = NeumannBC(location='right', value_fn=lambda t: 0.0)
    """

    def apply(self, u_grad: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Enforce du/dx = g(t) at boundary.

        Args:
            u_grad: Spatial gradient of solution at boundary
            t: Time values

        Returns:
            Residual: du/dx - g(t)
        """
        target_gradient = self.value_fn(t)
        return u_grad - target_gradient
