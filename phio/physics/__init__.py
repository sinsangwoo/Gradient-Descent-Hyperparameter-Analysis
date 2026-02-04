"""Physics problem definitions (PDEs).

This module provides functional API for physics-informed neural networks.
"""

from phio.physics.heat import (
    analytical_gaussian,
    dirichlet_bc_loss,
    heat_equation_residual,
    initial_condition_loss,
    steady_state_1d,
)

__all__ = [
    "heat_equation_residual",
    "analytical_gaussian",
    "steady_state_1d",
    "dirichlet_bc_loss",
    "initial_condition_loss",
]
