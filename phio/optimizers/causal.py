"""Causal weighting for time-dependent PDEs.

Implements the causal training strategy from Wang et al. (2022)
"When and why PINNs fail to train: A neural tangent kernel perspective"

Key idea: Learn solution progressively in time, respecting causality.
"""

from typing import Optional

import jax.numpy as jnp


class CausalWeightScheduler:
    """Adaptive causal weighting for time-dependent PDEs.

    For PDEs with time dimension, this scheduler gradually increases
    the temporal extent of training, respecting the causal structure
    of the PDE.

    Example:
        Epoch 0-1000:   Train on t ∈ [0, 0.2]
        Epoch 1000-2000: Train on t ∈ [0, 0.5]
        Epoch 2000+:     Train on t ∈ [0, 1.0]

    This prevents the network from trying to learn late-time behavior
    before it has mastered early-time dynamics.
    """

    def __init__(
        self,
        t_min: float = 0.0,
        t_max: float = 1.0,
        num_stages: int = 5,
        epochs_per_stage: int = 2000,
        epsilon: float = 0.1,
    ):
        """Initialize causal weight scheduler.

        Args:
            t_min: Minimum time in domain
            t_max: Maximum time in domain
            num_stages: Number of temporal stages to use
            epochs_per_stage: Epochs to train at each stage
            epsilon: Softness of temporal boundary (smooth transition)
        """
        self.t_min = t_min
        self.t_max = t_max
        self.num_stages = num_stages
        self.epochs_per_stage = epochs_per_stage
        self.epsilon = epsilon

        # Precompute stage boundaries
        self.stage_times = jnp.linspace(t_min, t_max, num_stages + 1)

    def get_temporal_weights(
        self, t: jnp.ndarray, epoch: int
    ) -> jnp.ndarray:
        """Compute causal weights for time points.

        Args:
            t: Time coordinates, shape (N,)
            epoch: Current training epoch

        Returns:
            weights: Temporal weights, shape (N,)
                     1.0 for t < t_cutoff(epoch)
                     Smooth decay for t ≈ t_cutoff
                     ≈0.0 for t > t_cutoff
        """
        # Determine current stage
        stage = min(epoch // self.epochs_per_stage, self.num_stages - 1)
        t_cutoff = self.stage_times[stage + 1]

        # Smooth sigmoid cutoff
        # w(t) = 1 / (1 + exp((t - t_cutoff) / epsilon))
        weights = 1.0 / (1.0 + jnp.exp((t - t_cutoff) / self.epsilon))

        return weights

    def get_stage_info(self, epoch: int) -> dict:
        """Get information about current training stage.

        Args:
            epoch: Current epoch

        Returns:
            Dictionary with stage info
        """
        stage = min(epoch // self.epochs_per_stage, self.num_stages - 1)
        t_cutoff = float(self.stage_times[stage + 1])

        return {
            "stage": stage,
            "t_min": float(self.t_min),
            "t_cutoff": t_cutoff,
            "t_max": float(self.t_max),
            "progress": (epoch % self.epochs_per_stage) / self.epochs_per_stage,
        }
