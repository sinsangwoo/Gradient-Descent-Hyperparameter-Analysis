"""Training callbacks for advanced features."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np


class Callback(ABC):
    """Base callback class.

    Callbacks allow custom logic at different stages of training.
    """

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of an epoch."""
        pass


class EarlyStoppingCallback(Callback):
    """Stop training when metric stops improving.

    Example:
        >>> callback = EarlyStoppingCallback(
        ...     monitor='val_loss',
        ...     patience=10,
        ...     min_delta=0.001
        ... )
    """

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        """Initialize early stopping.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
            verbose: Print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Reset state at training start."""
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check if training should stop."""
        if logs is None or self.monitor not in logs:
            return

        current = logs[self.monitor]

        # Check for improvement
        if self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta

        if improved:
            self.best_value = current
            self.wait = 0
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved to {current:.6f}")
        else:
            self.wait += 1
            if self.verbose:
                print(
                    f"Epoch {epoch}: {self.monitor} did not improve. "
                    f"Patience: {self.wait}/{self.patience}"
                )

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                if self.verbose:
                    print(
                        f"\nEarly stopping triggered at epoch {epoch}. "
                        f"Best {self.monitor}: {self.best_value:.6f}"
                    )


class CheckpointCallback(Callback):
    """Save checkpoints during training.

    Example:
        >>> callback = CheckpointCallback(
        ...     checkpoint_dir='checkpoints/',
        ...     save_freq=100,
        ...     save_best_only=True
        ... )
    """

    def __init__(
        self,
        checkpoint_manager,
        save_freq: int = 100,
        save_best_only: bool = False,
        monitor: str = "loss",
        mode: str = "min",
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_manager: CheckpointManager instance
            save_freq: Save every N epochs
            save_best_only: Only save when metric improves
            monitor: Metric to monitor for best checkpoint
            mode: 'min' or 'max'
        """
        self.checkpoint_manager = checkpoint_manager
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode

        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Save checkpoint if conditions are met."""
        if logs is None:
            return

        # Get state from logs
        state = logs.get("state")
        if state is None:
            return

        # Check if should save
        should_save = False

        if self.save_best_only:
            current = logs.get(self.monitor, float("inf"))
            if self.mode == "min":
                improved = current < self.best_value
            else:
                improved = current > self.best_value

            if improved:
                self.best_value = current
                should_save = True
        else:
            should_save = (epoch + 1) % self.save_freq == 0

        if should_save:
            metrics = {k: float(v) for k, v in logs.items() if k != "state"}
            self.checkpoint_manager.save(state, epoch, metrics)


class LearningRateScheduler(Callback):
    """Adjust learning rate during training.

    Example:
        >>> callback = LearningRateScheduler(
        ...     schedule=lambda epoch: 1e-3 * (0.95 ** epoch)
        ... )
    """

    def __init__(self, schedule, verbose: bool = True):
        """Initialize learning rate scheduler.

        Args:
            schedule: Function that takes epoch and returns learning rate
            verbose: Print messages
        """
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Update learning rate."""
        lr = self.schedule(epoch)

        if logs is not None and "state" in logs:
            # Update optimizer with new learning rate
            # This is a simplified version - actual implementation
            # would need to recreate optimizer
            if self.verbose:
                print(f"Epoch {epoch}: Learning rate = {lr:.6e}")


class TensorBoardCallback(Callback):
    """Log metrics to TensorBoard.

    Example:
        >>> callback = TensorBoardCallback(log_dir='logs/')
    """

    def __init__(self, log_dir: str = "logs"):
        """Initialize TensorBoard callback.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(self.log_dir))
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. " "Install with: pip install tensorboard")
            self.enabled = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log metrics to TensorBoard."""
        if not self.enabled or logs is None:
            return

        for key, value in logs.items():
            if key != "state" and isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

    def on_train_end(self, logs: Optional[Dict] = None):
        """Close TensorBoard writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()
