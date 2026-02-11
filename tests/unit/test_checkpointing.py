"""Unit tests for checkpointing."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from phio.training import CheckpointManager


class TestCheckpointManager:
    """Test CheckpointManager class."""

    def test_save_and_restore(self, tmp_path):
        """Test basic save/restore."""
        manager = CheckpointManager(str(tmp_path), max_to_keep=5)

        # Create dummy state
        state = SimpleNamespace(
            params={"w": jnp.array([1.0, 2.0])},
            opt_state={"step": 0},
        )

        # Save
        manager.save(state, epoch=100, metrics={"loss": 0.5})

        # Restore
        restored = manager.restore(state, epoch=100)

        assert jnp.allclose(restored.params["w"], state.params["w"])

    def test_max_to_keep(self, tmp_path):
        """Test max checkpoint limit."""
        manager = CheckpointManager(str(tmp_path), max_to_keep=3)

        state = SimpleNamespace(
            params={"w": jnp.array([1.0])}, opt_state={}
        )

        # Save 5 checkpoints
        for epoch in [10, 20, 30, 40, 50]:
            manager.save(state, epoch, {"loss": 1.0 / epoch})

        # Should keep only 3 best
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) <= 3

    def test_best_checkpoint(self, tmp_path):
        """Test best checkpoint selection."""
        manager = CheckpointManager(
            str(tmp_path), max_to_keep=5, keep_best=True
        )

        state = SimpleNamespace(
            params={"w": jnp.array([1.0])}, opt_state={}
        )

        # Save with different losses
        losses = {100: 1.0, 200: 0.5, 300: 0.3, 400: 0.6, 500: 0.4}
        for epoch, loss in losses.items():
            manager.save(state, epoch, {"loss": loss})

        # Best should be epoch 300 (loss=0.3)
        restored = manager.restore(state, best=True)
        assert restored is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
