"""Abstract base class for partial differential equations."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
import jax.numpy as jnp


class PDE(ABC):
    """Abstract base class for PDEs.

    All physics problems in PhIO inherit from this class and implement
    the residual computation method.

    Attributes:
        domain: Spatial domain as tuple (x_min, x_max) for 1D or dict for higher dims
        time_domain: Temporal domain as tuple (t_min, t_max)
        params: Dictionary of physical parameters (e.g., diffusion coefficient)
    """

    def __init__(
        self,
        domain: Any,
        time_domain: tuple = (0.0, 1.0),
        params: Dict[str, float] = None,
    ):
        """Initialize PDE.

        Args:
            domain: Spatial domain specification
            time_domain: Time interval (t_start, t_end)
            params: Physical parameters for the PDE
        """
        self.domain = domain
        self.time_domain = time_domain
        self.params = params or {}

    @abstractmethod
    def residual(self, u: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute PDE residual.

        The residual should be zero when u satisfies the PDE.

        Args:
            u: Solution values at points (x, t)
            x: Spatial coordinates
            t: Temporal coordinates

        Returns:
            PDE residual at each point

        Example:
            For heat equation: u_t - alpha * u_xx = 0
            residual = u_t - alpha * u_xx
        """
        pass

    @abstractmethod
    def exact_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Exact analytical solution (if available).

        Used for validation and benchmarking. Returns None if no exact solution exists.

        Args:
            x: Spatial coordinates
            t: Temporal coordinates

        Returns:
            Exact solution values at (x, t), or None
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(domain={self.domain}, time={self.time_domain})"
        )
