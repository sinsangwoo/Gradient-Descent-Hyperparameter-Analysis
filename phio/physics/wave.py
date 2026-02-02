"""Wave equation implementation."""

import jax.numpy as jnp
from phio.core import PDE


class WaveEquation1D(PDE):
    """1D Wave equation: u_tt = c^2 * u_xx.

    Models wave propagation in one spatial dimension.

    Args:
        domain: Spatial domain (x_min, x_max)
        time_domain: Time interval (t_start, t_end)
        wave_speed: Wave propagation speed c

    Example:
        >>> from phio.physics import WaveEquation1D
        >>> pde = WaveEquation1D(
        ...     domain=(0, 1),
        ...     time_domain=(0, 2),
        ...     wave_speed=1.0
        ... )
    """

    def __init__(
        self,
        domain: tuple = (0.0, 1.0),
        time_domain: tuple = (0.0, 1.0),
        wave_speed: float = 1.0,
    ):
        super().__init__(domain=domain, time_domain=time_domain)
        self.c = wave_speed

    def residual(self, u: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute wave equation residual: u_tt - c^2 * u_xx.

        Args:
            u: Solution values
            x: Spatial coordinates
            t: Temporal coordinates

        Returns:
            PDE residual
        """
        raise NotImplementedError(
            "Residual computation requires automatic differentiation in solver context"
        )

    def exact_solution(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Exact solution for d'Alembert's solution.

        For IC: u(x, 0) = sin(pi * x), u_t(x, 0) = 0
        And BC: u(0, t) = u(1, t) = 0

        Exact: u(x, t) = sin(pi * x) * cos(pi * c * t)

        Args:
            x: Spatial coordinates
            t: Temporal coordinates

        Returns:
            Exact solution values
        """
        return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * self.c * t)
