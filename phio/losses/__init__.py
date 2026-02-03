"""Loss functions for PINN training."""

from phio.losses.pinn_loss import pinn_loss, compute_pde_residual

__all__ = ["pinn_loss", "compute_pde_residual"]
