"""Core abstractions for PhIO framework."""

from phio.core.pde import PDE
from phio.core.boundary import BoundaryCondition, DirichletBC, NeumannBC
from phio.core.initial import InitialCondition

__all__ = [
    "PDE",
    "BoundaryCondition",
    "DirichletBC",
    "NeumannBC",
    "InitialCondition",
]
