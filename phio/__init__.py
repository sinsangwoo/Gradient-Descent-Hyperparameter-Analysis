"""PhIO: Physics-Informed Optimizer for solving PDEs with neural networks.

PhIO is a production-grade framework for solving partial differential equations
using Physics-Informed Neural Networks (PINNs). Built on JAX for GPU acceleration
and automatic differentiation.

Example:
    >>> import phio
    >>> from phio.physics import HeatEquation1D
    >>> from phio.solvers import PINNSolver
    >>>
    >>> # Define 1D heat equation
    >>> pde = HeatEquation1D(domain=(0, 1), diffusion_coeff=0.01)
    >>>
    >>> # Create solver
    >>> solver = PINNSolver(pde, hidden_dims=[64, 64, 64])
    >>>
    >>> # Train
    >>> results = solver.train(num_epochs=10000)
    >>> print(f"Final loss: {results['loss'][-1]:.2e}")
"""

__version__ = "0.1.0"
__author__ = "PhIO Contributors"
__license__ = "MIT"

from phio.core import (
    PDE,
    BoundaryCondition,
    InitialCondition,
)

from phio.solvers import (
    PINNSolver,
    AdaptivePINNSolver,
)

__all__ = [
    "PDE",
    "BoundaryCondition",
    "InitialCondition",
    "PINNSolver",
    "AdaptivePINNSolver",
]
