"""Multi-GPU and distributed training utilities."""

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


class DistributedTrainer:
    """Distributed training coordinator.

    Handles:
    - Multi-GPU data parallelism
    - Gradient synchronization
    - Device placement

    Example:
        >>> trainer = DistributedTrainer()
        >>> trainer.setup()  # Detect and configure GPUs
        >>> # Training automatically uses all available devices
    """

    def __init__(self):
        """Initialize distributed trainer."""
        self.n_devices = None
        self.devices = None

    def setup(self):
        """Setup distributed training."""
        self.n_devices = jax.device_count()
        self.devices = jax.devices()

        print("="*60)
        print("DISTRIBUTED TRAINING SETUP")
        print("="*60)
        print(f"Number of devices: {self.n_devices}")
        print(f"Device type: {self.devices[0].device_kind}")

        for i, device in enumerate(self.devices):
            print(f"  Device {i}: {device}")

        if self.n_devices == 1:
            print("\nWarning: Only 1 device available.")
            print("Multi-GPU features disabled.")
        else:
            print(f"\nData parallelism enabled across {self.n_devices} devices")

        print("="*60)

    def replicate_state(self, state: Any) -> Any:
        """Replicate state across all devices.

        Args:
            state: Training state

        Returns:
            Replicated state
        """
        if self.n_devices == 1:
            return state

        return jax.device_put_replicated(state, self.devices)

    def split_batch(self, batch: jnp.ndarray) -> jnp.ndarray:
        """Split batch across devices.

        Args:
            batch: Batch data

        Returns:
            Split batch (n_devices, batch_size // n_devices, ...)
        """
        if self.n_devices == 1:
            return batch

        # Reshape to (n_devices, -1, ...)
        return jnp.reshape(batch, (self.n_devices, -1) + batch.shape[1:])

    def create_pmap_step(
        self, step_fn: Callable
    ) -> Callable:
        """Create pmapped training step.

        Args:
            step_fn: Single-device step function

        Returns:
            Pmapped step function
        """
        if self.n_devices == 1:
            return step_fn

        @jax.pmap
        def pmap_step(state, batch):
            return step_fn(state, batch)

        return pmap_step

    def synchronize_gradients(
        self, grads: Any
    ) -> Any:
        """Average gradients across devices.

        Args:
            grads: Gradients from all devices

        Returns:
            Averaged gradients
        """
        if self.n_devices == 1:
            return grads

        return jax.lax.pmean(grads, axis_name="batch")

    def gather_metrics(
        self, metrics: jnp.ndarray
    ) -> float:
        """Gather and average metrics from all devices.

        Args:
            metrics: Metrics from all devices

        Returns:
            Averaged metric
        """
        if self.n_devices == 1:
            return float(metrics)

        return float(jnp.mean(metrics))


def estimate_speedup(n_devices: int, efficiency: float = 0.85) -> float:
    """Estimate speedup from multi-GPU training.

    Args:
        n_devices: Number of GPUs
        efficiency: Parallel efficiency (0-1)

    Returns:
        Expected speedup factor
    """
    # Amdahl's law with parallel efficiency
    return n_devices * efficiency


def print_device_info():
    """Print information about available devices."""
    n_devices = jax.device_count()
    devices = jax.devices()

    print("\nJAX Device Information:")
    print("="*60)
    print(f"Total devices: {n_devices}")
    print(f"Default backend: {jax.default_backend()}")

    for i, device in enumerate(devices):
        print(f"\nDevice {i}:")
        print(f"  Kind: {device.device_kind}")
        print(f"  ID: {device.id}")
        print(f"  Platform: {device.platform}")

    # Estimate speedup
    if n_devices > 1:
        speedup = estimate_speedup(n_devices)
        print(f"\nEstimated training speedup: {speedup:.1f}x")
        print(f"(assuming {0.85*100:.0f}% parallel efficiency)")

    print("="*60)
