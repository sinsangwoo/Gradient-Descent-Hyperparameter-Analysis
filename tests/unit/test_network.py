"""Test neural network architectures."""

import pytest
import jax
import jax.numpy as jnp
from jax import random
from phio.networks import MLP


class TestMLP:
    """Test multi-layer perceptron."""

    def test_initialization(self):
        """Test network can be created."""
        net = MLP(hidden_dims=[64, 64], output_dim=1, activation="tanh")
        assert net.hidden_dims == [64, 64]
        assert net.output_dim == 1

    def test_forward_pass_shape(self, rng):
        """Test forward pass produces correct output shape."""
        net = MLP(hidden_dims=[32, 32], output_dim=1)
        
        # Initialize
        x = random.normal(rng, (10, 2))  # Batch of 10, input dim 2
        params = net.init(rng, x)
        
        # Forward pass
        y = net.apply(params, x)
        assert y.shape == (10, 1)

    def test_different_activations(self, rng):
        """Test different activation functions."""
        activations = ["tanh", "relu", "gelu", "swish"]
        x = random.normal(rng, (5, 2))
        
        for act in activations:
            net = MLP(hidden_dims=[16], output_dim=1, activation=act)
            params = net.init(rng, x)
            y = net.apply(params, x)
            assert y.shape == (5, 1)

    def test_invalid_activation(self, rng):
        """Test error raised for invalid activation."""
        net = MLP(hidden_dims=[16], output_dim=1, activation="invalid")
        x = random.normal(rng, (5, 2))
        
        with pytest.raises(ValueError, match="Unknown activation"):
            params = net.init(rng, x)

    def test_gradient_flow(self, rng):
        """Test gradients can be computed."""
        net = MLP(hidden_dims=[32], output_dim=1)
        x = random.normal(rng, (10, 2))
        params = net.init(rng, x)
        
        def loss_fn(params):
            y = net.apply(params, x)
            return jnp.mean(y**2)
        
        # Compute gradients
        grads = jax.grad(loss_fn)(params)
        
        # Check gradients exist for all parameters
        assert grads is not None
        for leaf in jax.tree_util.tree_leaves(grads):
            assert not jnp.any(jnp.isnan(leaf))
