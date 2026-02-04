"""Advanced optimizers for Physics-Informed Neural Networks.

This module implements state-of-the-art optimization techniques specifically
designed for PINNs, addressing known challenges like gradient imbalance,
stiff PDEs, and multi-scale physics.
"""

from phio.optimizers.causal import CausalWeightScheduler
from phio.optimizers.loss_balancing import AdaptiveLossBalancer, NTKBalancer
from phio.optimizers.lbfgs import LBFGSOptimizer

__all__ = [
    "CausalWeightScheduler",
    "AdaptiveLossBalancer",
    "NTKBalancer",
    "LBFGSOptimizer",
]
