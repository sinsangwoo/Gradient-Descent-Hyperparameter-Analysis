"""Model checkpointing utilities."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp


class CheckpointManager:
    """Manage model checkpoints.

    Features:
    - Save/load model parameters
    - Save/load optimizer state
    - Keep best N checkpoints
    - Automatic cleanup of old checkpoints

    Example:
        >>> manager = CheckpointManager('checkpoints/', max_to_keep=3)
        >>> manager.save(state, epoch=100, metrics={'loss': 0.01})
        >>> state = manager.restore(epoch=100)
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_to_keep: int = 5,
        keep_best: bool = True,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep
            keep_best: Keep checkpoints with best metrics
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.checkpoints = []  # List of (epoch, path, metric) tuples

    def save(
        self,
        state: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save checkpoint.

        Args:
            state: Training state to save
            epoch: Current epoch
            metrics: Metrics to save (e.g., {'loss': 0.01})

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{epoch:06d}"
        checkpoint_path.mkdir(exist_ok=True)

        # Save parameters
        with open(checkpoint_path / "params.pkl", "wb") as f:
            pickle.dump(state.params, f)

        # Save optimizer state
        with open(checkpoint_path / "opt_state.pkl", "wb") as f:
            pickle.dump(state.opt_state, f)

        # Save metadata
        metadata = {
            "epoch": epoch,
            "metrics": metrics or {},
        }
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Track checkpoint
        metric_value = metrics.get("loss", float("inf")) if metrics else float("inf")
        self.checkpoints.append((epoch, checkpoint_path, metric_value))

        # Cleanup old checkpoints
        self._cleanup()

        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def restore(
        self,
        state: Any,
        epoch: Optional[int] = None,
        best: bool = False,
    ) -> Any:
        """Restore checkpoint.

        Args:
            state: Training state template
            epoch: Specific epoch to restore (None = latest)
            best: Restore best checkpoint instead of latest

        Returns:
            Restored training state
        """
        if best:
            checkpoint_path = self._get_best_checkpoint()
        elif epoch is not None:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{epoch:06d}"
        else:
            checkpoint_path = self._get_latest_checkpoint()

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load parameters
        with open(checkpoint_path / "params.pkl", "rb") as f:
            params = pickle.load(f)

        # Load optimizer state
        with open(checkpoint_path / "opt_state.pkl", "rb") as f:
            opt_state = pickle.load(f)

        # Load metadata
        with open(checkpoint_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Update state
        state = state.replace(params=params, opt_state=opt_state)

        print(f"Checkpoint restored: {checkpoint_path}")
        print(f"Epoch: {metadata['epoch']}, Metrics: {metadata['metrics']}")

        return state

    def _get_latest_checkpoint(self) -> Path:
        """Get path to latest checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        return checkpoints[-1]

    def _get_best_checkpoint(self) -> Path:
        """Get path to checkpoint with best metric."""
        if not self.checkpoints:
            raise FileNotFoundError("No checkpoints tracked")

        # Sort by metric (lower is better)
        best = min(self.checkpoints, key=lambda x: x[2])
        return best[1]

    def _cleanup(self):
        """Remove old checkpoints beyond max_to_keep."""
        if len(self.checkpoints) <= self.max_to_keep:
            return

        if self.keep_best:
            # Sort by metric and keep best N
            self.checkpoints.sort(key=lambda x: x[2])
            to_remove = self.checkpoints[self.max_to_keep :]
        else:
            # Keep latest N
            self.checkpoints.sort(key=lambda x: x[0])
            to_remove = self.checkpoints[: -self.max_to_keep]

        # Delete checkpoints
        for epoch, path, _ in to_remove:
            if path.exists():
                import shutil

                shutil.rmtree(path)
                print(f"Removed old checkpoint: {path}")

        # Update tracked checkpoints
        self.checkpoints = self.checkpoints[-self.max_to_keep :]

    def list_checkpoints(self) -> list:
        """List all available checkpoints.

        Returns:
            List of (epoch, path, metric) tuples
        """
        return sorted(self.checkpoints, key=lambda x: x[0])
