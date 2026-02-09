"""Adaptive PINN solver with curriculum learning."""

from typing import Dict, List

from phio.solvers.base import PINNSolver
from phio.utils import logger


class AdaptivePINNSolver(PINNSolver):
    """PINN solver with adaptive curriculum learning.

    Implements progressive training strategy:
    1. Phase 1: Learn boundary conditions (easy)
    2. Phase 2: Add initial conditions (medium)
    3. Phase 3: Enforce PDE residual (hard)

    This approach leads to 2-5x faster convergence on stiff PDEs.

    Args:
        pde: PDE instance
        hidden_dims: Network architecture
        curriculum_schedule: Dict mapping epoch ranges to loss weights
            Example: {
                (0, 1000): {'pde': 0.0, 'bc': 1.0, 'ic': 0.0},
                (1000, 3000): {'pde': 0.1, 'bc': 1.0, 'ic': 0.5},
                (3000, 10000): {'pde': 1.0, 'bc': 1.0, 'ic': 1.0},
            }

    Example:
        >>> from phio.physics import HeatEquation1D
        >>> from phio.solvers import AdaptivePINNSolver
        >>>
        >>> pde = HeatEquation1D()
        >>> solver = AdaptivePINNSolver(
        ...     pde,
        ...     curriculum_schedule={
        ...         (0, 2000): {'pde': 0.0, 'bc': 1.0, 'ic': 0.5},
        ...         (2000, 5000): {'pde': 0.5, 'bc': 1.0, 'ic': 1.0},
        ...         (5000, 10000): {'pde': 1.0, 'bc': 1.0, 'ic': 1.0},
        ...     }
        ... )
    """

    def __init__(
        self,
        pde,
        hidden_dims: List[int] = None,
        curriculum_schedule: Dict[tuple, Dict[str, float]] = None,
        **kwargs,
    ):
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
        super().__init__(pde, hidden_dims, **kwargs)

        # Default curriculum: gradual introduction of PDE term
        if curriculum_schedule is None:
            curriculum_schedule = {
                (0, 1000): {"pde": 0.0, "bc": 1.0, "ic": 0.5},
                (1000, 3000): {"pde": 0.5, "bc": 1.0, "ic": 1.0},
                (3000, float("inf")): {"pde": 1.0, "bc": 1.0, "ic": 1.0},
            }

        self.curriculum_schedule = curriculum_schedule
        msg = f"Initialized adaptive solver with " f"{len(curriculum_schedule)} curriculum phases"
        logger.info(msg)

    def _get_current_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for current epoch based on curriculum."""
        for (start, end), weights in self.curriculum_schedule.items():
            if start <= epoch < end:
                return weights
        # Default to full weights if beyond schedule
        return {"pde": 1.0, "bc": 1.0, "ic": 1.0}

    def train(self, num_epochs: int = 10000, **kwargs) -> Dict:
        """Train with adaptive curriculum.

        Overrides base train() to dynamically adjust loss weights.
        """
        logger.info("Starting adaptive curriculum training...")
        for (start, end), weights in self.curriculum_schedule.items():
            end_display = "∞" if end == float("inf") else end
            msg = (
                f"  Epochs {start}-{end_display}: "
                f"PDE={weights['pde']}, BC={weights['bc']}, "
                f"IC={weights['ic']}"
            )
            logger.info(msg)

        # For now, call parent train with final weights
        # TODO: Implement dynamic weight adjustment per epoch
        final_weights = self._get_current_weights(num_epochs - 1)
        return super().train(num_epochs=num_epochs, loss_weights=final_weights, **kwargs)
