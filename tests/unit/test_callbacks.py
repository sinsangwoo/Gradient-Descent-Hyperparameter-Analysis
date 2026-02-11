"""Unit tests for training callbacks."""

import pytest

from phio.training import EarlyStoppingCallback, LearningRateScheduler


class TestEarlyStoppingCallback:
    """Test EarlyStoppingCallback."""

    def test_early_stopping_triggered(self):
        """Test that early stopping triggers."""
        callback = EarlyStoppingCallback(
            monitor="loss", patience=3, min_delta=0.001, mode="min"
        )

        callback.on_train_begin()

        # Simulate plateau
        losses = [1.0, 0.5, 0.3, 0.3, 0.3, 0.3]
        for epoch, loss in enumerate(losses):
            callback.on_epoch_end(epoch, {"loss": loss})

        assert callback.should_stop
        assert callback.wait >= callback.patience

    def test_improvement_resets_patience(self):
        """Test that improvement resets patience."""
        callback = EarlyStoppingCallback(
            monitor="loss", patience=3, min_delta=0.001, mode="min"
        )

        callback.on_train_begin()

        # Improving then plateau
        losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.7]  # Improves at end
        for epoch, loss in enumerate(losses):
            callback.on_epoch_end(epoch, {"loss": loss})

        assert not callback.should_stop
        assert callback.wait < callback.patience

    def test_mode_max(self):
        """Test max mode (higher is better)."""
        callback = EarlyStoppingCallback(
            monitor="accuracy", patience=3, mode="max"
        )

        callback.on_train_begin()

        # Accuracy stops improving
        accuracies = [0.5, 0.7, 0.9, 0.9, 0.9, 0.9]
        for epoch, acc in enumerate(accuracies):
            callback.on_epoch_end(epoch, {"accuracy": acc})

        assert callback.should_stop


class TestLearningRateScheduler:
    """Test LearningRateScheduler."""

    def test_schedule_called(self):
        """Test that schedule function is called."""
        called_epochs = []

        def schedule(epoch):
            called_epochs.append(epoch)
            return 1e-3 * (0.9 ** epoch)

        callback = LearningRateScheduler(schedule, verbose=False)

        for epoch in range(5):
            callback.on_epoch_begin(epoch)

        assert called_epochs == [0, 1, 2, 3, 4]

    def test_exponential_decay(self):
        """Test exponential decay schedule."""
        def schedule(epoch):
            return 1e-3 * (0.9 ** epoch)

        callback = LearningRateScheduler(schedule, verbose=False)

        lr_0 = schedule(0)
        lr_10 = schedule(10)

        assert lr_0 == 1e-3
        assert lr_10 < lr_0
        assert lr_10 == pytest.approx(1e-3 * (0.9 ** 10))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
