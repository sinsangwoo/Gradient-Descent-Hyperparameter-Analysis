"""Multi-layer perceptron (MLP) implementation using Flax."""

from typing import List, Callable
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    """Multi-layer perceptron for PINN approximation.

    Standard feedforward neural network with configurable depth and width.

    Attributes:
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (typically 1 for scalar PDEs)
        activation: Activation function ('tanh', 'relu', 'gelu', 'swish')

    Example:
        >>> import jax
        >>> from phio.networks import MLP
        >>>
        >>> # Create 3-layer network: [2] -> [64] -> [64] -> [64] -> [1]
        >>> net = MLP(hidden_dims=[64, 64, 64], output_dim=1, activation='tanh')
        >>>
        >>> # Initialize
        >>> rng = jax.random.PRNGKey(0)
        >>> x = jax.random.normal(rng, (10, 2))  # Batch of 10, input dim 2
        >>> params = net.init(rng, x)
        >>>
        >>> # Forward pass
        >>> y = net.apply(params, x)
        >>> print(y.shape)  # (10, 1)
    """

    hidden_dims: List[int]
    output_dim: int = 1
    activation: str = "tanh"

    def setup(self):
        """Initialize layers."""
        # Map activation name to function
        activation_map = {
            "tanh": nn.tanh,
            "relu": nn.relu,
            "gelu": nn.gelu,
            "swish": nn.swish,
            "sigmoid": nn.sigmoid,
        }
        if self.activation not in activation_map:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Choose from: {list(activation_map.keys())}"
            )
        self.act_fn = activation_map[self.activation]

        # Create hidden layers
        self.hidden_layers = [
            nn.Dense(features=dim, name=f"hidden_{i}")
            for i, dim in enumerate(self.hidden_dims)
        ]

        # Output layer
        self.output_layer = nn.Dense(features=self.output_dim, name="output")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Forward through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.act_fn(x)

        # Output layer (no activation)
        x = self.output_layer(x)

        return x

    def __repr__(self) -> str:
        """String representation."""
        arch = " -> ".join(["input"] + [str(d) for d in self.hidden_dims] + [str(self.output_dim)])
        return f"MLP({arch}, activation={self.activation})"
