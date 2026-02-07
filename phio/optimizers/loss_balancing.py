"""Adaptive loss balancing for multi-objective PINN training.

Implements various strategies for balancing PDE, BC, and IC losses:
1. GradNorm (Chen et al. 2018)
2. NTK-based balancing (Wang et al. 2021)
3. Adaptive weighting with exponential moving average
"""

from typing import Dict, Optional

import jax
import jax.numpy as jnp


class AdaptiveLossBalancer:
    """Adaptive loss balancing using gradient magnitude balancing.

    Automatically adjusts loss weights to balance gradient magnitudes
    across different loss components (PDE, BC, IC).

    Based on GradNorm (Chen et al. 2018) but simplified for PINNs.
    """

    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        alpha: float = 0.1,
        update_freq: int = 100,
    ):
        """Initialize adaptive loss balancer.

        Args:
            initial_weights: Initial loss weights {"pde": 1.0, "bc": 1.0, "ic": 1.0}
            alpha: Learning rate for weight updates
            update_freq: Update weights every N epochs
        """
        if initial_weights is None:
            initial_weights = {"pde": 1.0, "bc": 1.0, "ic": 1.0}

        self.weights = initial_weights.copy()
        self.alpha = alpha
        self.update_freq = update_freq

        # Track gradient history for stability
        self.grad_history = {k: [] for k in self.weights.keys()}

    def update_weights(
        self, gradients: Dict[str, jnp.ndarray], epoch: int
    ) -> Dict[str, float]:
        """Update loss weights based on gradient magnitudes.

        Args:
            gradients: Dict of gradients for each loss component
            epoch: Current epoch

        Returns:
            Updated weights
        """
        if epoch % self.update_freq != 0:
            return self.weights

        # Compute gradient magnitudes (L2 norm)
        grad_norms = {}
        for key, grad in gradients.items():
            # Flatten gradient and compute norm
            flat_grad = jax.tree_util.tree_leaves(grad)
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in flat_grad))
            grad_norms[key] = float(grad_norm)

        # Target: All gradient norms should be equal
        mean_grad = jnp.mean(jnp.array(list(grad_norms.values())))

        # Update weights to balance gradients
        for key in self.weights.keys():
            if grad_norms[key] > 0:
                # If gradient too large, decrease weight
                # If gradient too small, increase weight
                ratio = mean_grad / (grad_norms[key] + 1e-8)
                self.weights[key] *= 1.0 + self.alpha * (ratio - 1.0)

                # Clamp weights to reasonable range
                self.weights[key] = float(jnp.clip(self.weights[key], 0.01, 100.0))

        return self.weights


class NTKBalancer:
    """Neural Tangent Kernel based loss balancing.

    Uses NTK eigenvalue analysis to balance loss components.
    Based on Wang et al. (2021) Understanding and mitigating gradient
    pathologies in PINNs.

    Key insight: Loss components with larger NTK eigenvalues dominate
    training. Reweight to equalize influence.
    """

    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        update_freq: int = 500,
        ema_decay: float = 0.9,
    ):
        """Initialize NTK-based balancer.

        Args:
            initial_weights: Initial weights
            update_freq: Recompute NTK every N epochs
            ema_decay: Exponential moving average decay for stability
        """
        if initial_weights is None:
            initial_weights = {"pde": 1.0, "bc": 1.0, "ic": 1.0}

        self.weights = initial_weights.copy()
        self.update_freq = update_freq
        self.ema_decay = ema_decay

        # Track NTK eigenvalues
        self.ntk_eigenvalues = {k: 1.0 for k in self.weights.keys()}

    def compute_ntk_eigenvalue(
        self, loss_fn, params: dict, sample_points: jnp.ndarray
    ) -> float:
        """Estimate largest NTK eigenvalue for a loss component.

        Args:
            loss_fn: Loss function
            params: Model parameters
            sample_points: Sample points for NTK computation

        Returns:
            Approximate largest eigenvalue
        """
        # Compute Jacobian of loss w.r.t. parameters
        jacobian = jax.jacrev(loss_fn)(params)

        # Flatten Jacobian
        flat_jac = jax.tree_util.tree_leaves(jacobian)
        J = jnp.concatenate([g.flatten() for g in flat_jac])

        # NTK ≈ J^T J, largest eigenvalue ≈ ||J||^2
        eigenvalue = float(jnp.sum(J**2))

        return eigenvalue

    def update_weights_ntk(
        self,
        ntk_eigenvalues: Dict[str, float],
        epoch: int,
    ) -> Dict[str, float]:
        """Update weights based on NTK eigenvalues.

        Args:
            ntk_eigenvalues: Dict of NTK eigenvalues for each loss
            epoch: Current epoch

        Returns:
            Updated weights
        """
        if epoch % self.update_freq != 0:
            return self.weights

        # Update eigenvalue estimates with EMA
        for key, eig in ntk_eigenvalues.items():
            self.ntk_eigenvalues[key] = (
                self.ema_decay * self.ntk_eigenvalues[key] + (1 - self.ema_decay) * eig
            )

        # Target: Equalize effective NTK influence
        # Effective influence = weight * eigenvalue
        # Want: weight_i * eig_i ≈ constant
        mean_eig = jnp.mean(jnp.array(list(self.ntk_eigenvalues.values())))

        for key in self.weights.keys():
            if self.ntk_eigenvalues[key] > 0:
                # Inverse weighting: larger eigenvalue → smaller weight
                self.weights[key] = float(mean_eig / (self.ntk_eigenvalues[key] + 1e-8))

        # Normalize weights
        total_weight = sum(self.weights.values())
        for key in self.weights.keys():
            self.weights[key] /= total_weight
            self.weights[key] *= 3.0  # Rescale to sum to 3.0

        return self.weights
