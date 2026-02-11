"""Advanced trainer with callbacks and multi-device support."""

from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
from tqdm import tqdm

from phio.training.callbacks import Callback


class AdvancedTrainer:
    """Advanced PINN trainer with callbacks and multi-device support.

    Features:
    - Callback system (early stopping, checkpointing)
    - Multi-GPU/TPU support via JAX pmap
    - Progress bars with tqdm
    - Flexible loss computation
    - History tracking

    Example:
        >>> trainer = AdvancedTrainer(
        ...     state=state,
        ...     loss_fn=compute_loss,
        ...     callbacks=[early_stopping, checkpoint_callback]
        ... )
        >>> history = trainer.train(num_epochs=5000)
    """

    def __init__(
        self,
        state,
        loss_fn: Callable,
        callbacks: Optional[List[Callback]] = None,
        use_pmap: bool = False,
    ):
        """Initialize trainer.

        Args:
            state: Training state
            loss_fn: Loss computation function
            callbacks: List of callbacks
            use_pmap: Use pmap for multi-device training
        """
        self.state = state
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []
        self.use_pmap = use_pmap
        self.history = {"loss": [], "epoch": []}

        # Setup multi-device if requested
        if use_pmap:
            self._setup_pmap()

    def _setup_pmap(self):
        """Setup parallel map for multi-device training."""
        self.n_devices = jax.device_count()
        print(f"Setting up pmap for {self.n_devices} devices")

        if self.n_devices == 1:
            print("Warning: Only 1 device available, pmap disabled")
            self.use_pmap = False
            return

        # Replicate state across devices
        self.state = jax.device_put_replicated(self.state, jax.devices())

    def train(
        self,
        train_data: Dict,
        num_epochs: int,
        validation_data: Optional[Dict] = None,
        print_every: int = 100,
    ) -> Dict:
        """Train model.

        Args:
            train_data: Training data dictionary
            num_epochs: Number of epochs
            validation_data: Optional validation data
            print_every: Print frequency

        Returns:
            Training history
        """
        # Call on_train_begin callbacks
        for callback in self.callbacks:
            callback.on_train_begin()

        # Training loop
        with tqdm(range(num_epochs), desc="Training") as pbar:
            for epoch in pbar:
                # Call on_epoch_begin callbacks
                logs = {"epoch": epoch, "state": self.state}
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch, logs)

                # Training step
                self.state, loss = self._train_step(train_data)

                # Validation step
                val_loss = None
                if validation_data is not None:
                    val_loss = self._validation_step(validation_data)

                # Update history
                self.history["loss"].append(float(loss))
                self.history["epoch"].append(epoch)
                if val_loss is not None:
                    if "val_loss" not in self.history:
                        self.history["val_loss"] = []
                    self.history["val_loss"].append(float(val_loss))

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss:.6e}"})

                # Print progress
                if (epoch + 1) % print_every == 0:
                    msg = f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6e}"
                    if val_loss is not None:
                        msg += f", Val Loss: {val_loss:.6e}"
                    print(msg)

                # Call on_epoch_end callbacks
                logs = {
                    "epoch": epoch,
                    "loss": float(loss),
                    "state": self.state,
                }
                if val_loss is not None:
                    logs["val_loss"] = float(val_loss)

                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, logs)

                # Check early stopping
                should_stop = any(
                    getattr(cb, "should_stop", False) for cb in self.callbacks
                )
                if should_stop:
                    print(f"\nTraining stopped early at epoch {epoch+1}")
                    break

        # Call on_train_end callbacks
        for callback in self.callbacks:
            callback.on_train_end({"history": self.history})

        return self.history

    def _train_step(self, data: Dict):
        """Single training step.

        Args:
            data: Training data

        Returns:
            Updated state and loss
        """
        if self.use_pmap:
            return self._train_step_pmap(data)
        else:
            return self._train_step_single(data)

    def _train_step_single(self, data: Dict):
        """Single device training step."""

        def loss_and_grad(params):
            loss = self.loss_fn(params, data)
            return loss, loss

        grad_fn = jax.value_and_grad(loss_and_grad, has_aux=True)
        (loss, _), grads = grad_fn(self.state.params)

        # Update parameters
        updates, opt_state = self.state.tx.update(
            grads, self.state.opt_state, self.state.params
        )
        params = jax.tree_map(lambda p, u: p + u, self.state.params, updates)

        # Update state
        self.state = self.state.replace(
            step=self.state.step + 1, params=params, opt_state=opt_state
        )

        return self.state, loss

    def _train_step_pmap(self, data: Dict):
        """Multi-device training step with pmap."""
        # Split data across devices
        data_split = jax.tree_map(
            lambda x: jnp.reshape(x, (self.n_devices, -1) + x.shape[1:]), data
        )

        # Define pmapped training step
        @jax.pmap
        def pmap_step(state, batch):
            def loss_and_grad(params):
                loss = self.loss_fn(params, batch)
                return loss, loss

            grad_fn = jax.value_and_grad(loss_and_grad, has_aux=True)
            (loss, _), grads = grad_fn(state.params)

            # Average gradients across devices
            grads = jax.lax.pmean(grads, axis_name="batch")

            # Update parameters
            updates, opt_state = state.tx.update(
                grads, state.opt_state, state.params
            )
            params = jax.tree_map(lambda p, u: p + u, state.params, updates)

            return (
                state.replace(step=state.step + 1, params=params, opt_state=opt_state),
                loss,
            )

        # Execute on all devices
        self.state, losses = pmap_step(self.state, data_split)

        # Average loss across devices
        loss = jnp.mean(losses)

        return self.state, loss

    def _validation_step(self, data: Dict) -> float:
        """Validation step.

        Args:
            data: Validation data

        Returns:
            Validation loss
        """
        if self.use_pmap:
            # Get parameters from first device
            params = jax.tree_map(lambda x: x[0], self.state.params)
        else:
            params = self.state.params

        loss = self.loss_fn(params, data)
        return float(loss)

    def get_state(self):
        """Get current training state.

        Returns:
            Training state (unreplicated if using pmap)
        """
        if self.use_pmap:
            # Get state from first device
            return jax.tree_map(lambda x: x[0], self.state)
        else:
            return self.state
