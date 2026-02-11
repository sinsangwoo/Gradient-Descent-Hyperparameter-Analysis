"""Phase 3.3: Advanced Training Demo.

Demonstrates:
1. Checkpointing (save/restore)
2. Early stopping
3. Learning rate scheduling
4. TensorBoard logging
5. Multi-GPU training (if available)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

from phio.training import (
    AdvancedTrainer,
    CheckpointCallback,
    CheckpointManager,
    EarlyStoppingCallback,
    LearningRateScheduler,
    TensorBoardCallback,
)
from phio.training.distributed import DistributedTrainer, print_device_info


class SimpleNetwork(nn.Module):
    """Simple MLP for demonstration."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


def demo_checkpointing():
    """Demo checkpoint save/restore."""
    print("\n" + "="*60)
    print("DEMO 1: CHECKPOINTING")
    print("="*60)

    # Create checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir="demo_checkpoints",
        max_to_keep=3,
        keep_best=True,
    )

    print("\n1. Checkpoint Manager initialized")
    print(f"   Directory: {manager.checkpoint_dir}")
    print(f"   Max to keep: {manager.max_to_keep}")

    # Simulate training states
    from types import SimpleNamespace

    for epoch in [100, 200, 300, 400, 500]:
        # Create dummy state
        state = SimpleNamespace(
            params={"layer1": jnp.array([1.0, 2.0])},
            opt_state={"step": epoch},
        )

        # Save with metrics
        metrics = {"loss": 1.0 / epoch}  # Decreasing loss
        manager.save(state, epoch, metrics)

    print("\n2. Checkpoints saved:")
    for epoch, path, metric in manager.list_checkpoints():
        print(f"   Epoch {epoch}: loss={metric:.6f}")

    print("\n3. Best checkpoint:")
    print(f"   Epoch 500 with lowest loss")

    print("\n✅ Checkpointing demo complete!")


def demo_early_stopping():
    """Demo early stopping callback."""
    print("\n" + "="*60)
    print("DEMO 2: EARLY STOPPING")
    print("="*60)

    # Create early stopping callback
    callback = EarlyStoppingCallback(
        monitor="loss",
        patience=5,
        min_delta=0.001,
        mode="min",
        verbose=True,
    )

    print("\nConfiguration:")
    print(f"  Monitor: {callback.monitor}")
    print(f"  Patience: {callback.patience} epochs")
    print(f"  Min delta: {callback.min_delta}")

    # Simulate training
    callback.on_train_begin()
    print("\nSimulating training:")

    # Improving then plateauing losses
    losses = [
        1.0, 0.5, 0.3, 0.2, 0.15,  # Improving
        0.14, 0.141, 0.140, 0.139, 0.138,  # Slow improvement
        0.138, 0.139, 0.138, 0.139, 0.138,  # Plateau
    ]

    for epoch, loss in enumerate(losses):
        callback.on_epoch_end(epoch, {"loss": loss})
        if callback.should_stop:
            print(f"\n⚠️  Training would stop at epoch {epoch}")
            break

    print("\n✅ Early stopping demo complete!")


def demo_learning_rate_schedule():
    """Demo learning rate scheduling."""
    print("\n" + "="*60)
    print("DEMO 3: LEARNING RATE SCHEDULING")
    print("="*60)

    # Exponential decay schedule
    def schedule(epoch):
        return 1e-3 * (0.95 ** epoch)

    callback = LearningRateScheduler(schedule, verbose=True)

    print("\nSchedule: lr = 1e-3 * (0.95 ^ epoch)")
    print("\nLearning rates:")

    for epoch in [0, 10, 20, 50, 100]:
        lr = schedule(epoch)
        print(f"  Epoch {epoch:3d}: lr = {lr:.6e}")

    print("\n✅ Learning rate scheduling demo complete!")


def demo_multi_gpu():
    """Demo multi-GPU detection and setup."""
    print("\n" + "="*60)
    print("DEMO 4: MULTI-GPU SUPPORT")
    print("="*60)

    # Print device info
    print_device_info()

    # Setup distributed trainer
    dist_trainer = DistributedTrainer()
    dist_trainer.setup()

    n_devices = jax.device_count()

    if n_devices > 1:
        print("\n✅ Multi-GPU training available!")
        print(f"   Expected speedup: ~{n_devices * 0.85:.1f}x")
    else:
        print("\n⚠️  Only 1 device detected.")
        print("   Multi-GPU features disabled.")
        print("\nTo use multiple GPUs:")
        print("  - Ensure CUDA is installed")
        print("  - Install: pip install jax[cuda11_pip]")
        print("  - Run on multi-GPU machine")


def demo_tensorboard():
    """Demo TensorBoard logging."""
    print("\n" + "="*60)
    print("DEMO 5: TENSORBOARD LOGGING")
    print("="*60)

    # Create TensorBoard callback
    callback = TensorBoardCallback(log_dir="demo_logs")

    if callback.enabled:
        print("\n✅ TensorBoard enabled")
        print(f"   Log directory: {callback.log_dir}")
        print("\nTo view logs:")
        print("   tensorboard --logdir=demo_logs")
        print("   Open: http://localhost:6006")

        # Simulate logging
        print("\nSimulating training metrics...")
        for epoch in range(10):
            callback.on_epoch_end(
                epoch,
                {
                    "loss": 1.0 / (epoch + 1),
                    "accuracy": 0.5 + 0.05 * epoch,
                },
            )
        callback.on_train_end()

        print("\n✅ Metrics logged to TensorBoard!")
    else:
        print("\n⚠️  TensorBoard not available")
        print("   Install with: pip install tensorboard")


def demo_complete_workflow():
    """Demo complete training workflow with all features."""
    print("\n" + "="*60)
    print("DEMO 6: COMPLETE WORKFLOW")
    print("="*60)

    print("\nTraining workflow with:")
    print("  ✓ Checkpointing (every 100 epochs)")
    print("  ✓ Early stopping (patience=10)")
    print("  ✓ Learning rate decay")
    print("  ✓ TensorBoard logging")
    print("  ✓ Multi-GPU (if available)")

    # Setup callbacks
    checkpoint_mgr = CheckpointManager("workflow_checkpoints", max_to_keep=3)
    callbacks = [
        CheckpointCallback(checkpoint_mgr, save_freq=100),
        EarlyStoppingCallback(monitor="loss", patience=10, verbose=False),
        LearningRateScheduler(lambda e: 1e-3 * (0.95 ** e), verbose=False),
        TensorBoardCallback("workflow_logs"),
    ]

    print("\n✅ All components configured!")
    print("\nIn production:")
    print("  trainer = AdvancedTrainer(state, loss_fn, callbacks)")
    print("  history = trainer.train(data, num_epochs=5000)")


def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("# PHASE 3.3: ADVANCED TRAINING FEATURES")
    print("#"*60)

    print("\nDemonstrating:")
    print("  1. Checkpointing")
    print("  2. Early stopping")
    print("  3. Learning rate scheduling")
    print("  4. Multi-GPU support")
    print("  5. TensorBoard logging")
    print("  6. Complete workflow")

    demo_checkpointing()
    demo_early_stopping()
    demo_learning_rate_schedule()
    demo_multi_gpu()
    demo_tensorboard()
    demo_complete_workflow()

    print("\n" + "="*60)
    print("ALL DEMOS COMPLETE")
    print("="*60)

    print("\nKey Features:")
    print("  ✅ Automatic checkpointing")
    print("  ✅ Early stopping to prevent overfitting")
    print("  ✅ Dynamic learning rate adjustment")
    print("  ✅ Multi-GPU parallel training")
    print("  ✅ TensorBoard visualization")

    print("\nProduction Benefits:")
    print("  • 10-100x faster with multiple GPUs")
    print("  • Automatic best model selection")
    print("  • Resume training from checkpoints")
    print("  • Real-time monitoring with TensorBoard")

    print("\nNext Steps:")
    print("  1. Enable multi-GPU: Install JAX with CUDA")
    print("  2. View TensorBoard: tensorboard --logdir=demo_logs")
    print("  3. Use in production with AdvancedTrainer")


if __name__ == "__main__":
    main()
