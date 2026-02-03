"""Heat equation (diffusion) implementation."""

import jax
import jax.numpy as jnp
from phio.core import PDE


class HeatEquation1D(PDE):
    """1D Heat equation: u_t = alpha * u_xx.

    Models heat diffusion in one spatial dimension.

    Args:
        domain: Spatial domain (x_min, x_max)
        time_domain: Time interval (t_start, t_end)
        diffusion_coeff: Thermal diffusivity alpha

    Example:
        >>> from phio.physics import HeatEquation1D
        >>> pde = HeatEquation1D(
        ...     domain=(0, 1),
        ...     time_domain=(0, 1),
        ...     diffusion_coeff=0.01
        ... )
    """

    def __init__(
        self,
        domain: tuple = (0.0, 1.0),
        time_domain: tuple = (0.0, 1.0),
        diffusion_coeff: float = 0.01,
    ):
        super().__init__(domain=domain, time_domain=time_domain)
        self.alpha = diffusion_coeff

    def residual(self, u: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute heat equation residual: u_t - alpha * u_xx.

        Args:
            u: Solution values (must be differentiable w.r.t x and t)
            x: Spatial coordinates
            t: Temporal coordinates

        Returns:
            PDE residual at each point (should be ~0 for valid solution)
        """
        # This will be implemented with automatic differentiation in the solver
        # For now, this is a placeholder showing the mathematical form
        raise NotImplementedError(
            "Residual computation requires automatic differentiation in solver context"
        )

    def exact_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Exact solution for specific initial/boundary conditions.

        For IC: u(x, 0) = sin(pi * x)
        And BC: u(0, t) = u(1, t) = 0

        Exact solution: u(x, t) = exp(-alpha * pi^2 * t) * sin(pi * x)

        Args:
            x: Spatial coordinates
            t: Temporal coordinates

        Returns:
            Exact solution values
        """
        return jnp.exp(-self.alpha * jnp.pi**2 * t) * jnp.sin(jnp.pi * x)

    def __repr__(self) -> str:
        return (
            f"HeatEquation1D(domain={self.domain}, "
            f"alpha={self.alpha}, time={self.time_domain})"
        )
