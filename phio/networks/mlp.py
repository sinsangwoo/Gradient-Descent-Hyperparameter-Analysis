"""Multi-layer perceptron architectures for PINNs."""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    """Standard fully-connected network for PINN.

    Attributes:
        features: Sequence of hidden layer sizes, e.g. [64, 64, 64]
        activation: Activation function (default: tanh)
        use_bias: Whether to include bias terms
    """

    features: Sequence[int] = (64, 64, 64, 1)
    activation: callable = nn.tanh
    use_bias: bool = True

    @nn.compact
    def __call__(self, x, t):
        """Forward pass.

        Args:
            x: Spatial coordinate(s), shape (..., spatial_dim)
            t: Time coordinate, shape (..., 1)

        Returns:
            Solution u(x, t), shape (..., 1)
        """
        # Concatenate space and time
        inputs = jnp.concatenate([x, t], axis=-1)

        # Hidden layers
        z = inputs
        for feat in self.features[:-1]:
            z = nn.Dense(feat, use_bias=self.use_bias)(z)
            z = self.activation(z)

        # Output layer (no activation)
        output = nn.Dense(self.features[-1], use_bias=self.use_bias)(z)
        return output


class ResidualMLP(nn.Module):
    """MLP with residual connections for deeper networks.

    Residual connections help gradient flow in deeper PINNs,
    addressing the "spectral bias" issue where networks struggle
    to learn high-frequency components.
    """

    features: Sequence[int] = (64, 64, 64, 1)
    activation: callable = nn.tanh
    use_bias: bool = True

    @nn.compact
    def __call__(self, x, t):
        inputs = jnp.concatenate([x, t], axis=-1)

        z = inputs
        for i, feat in enumerate(self.features[:-1]):
            z_new = nn.Dense(feat, use_bias=self.use_bias)(z)
            z_new = self.activation(z_new)

            # Add residual connection if dimensions match
            if z.shape[-1] == feat and i > 0:
                z = z + z_new  # Residual
            else:
                z = z_new

        # Output layer
        output = nn.Dense(self.features[-1], use_bias=self.use_bias)(z)
        return output


class FourierFeatureMLP(nn.Module):
    """MLP with Fourier feature encoding for better high-frequency learning.

    References:
        Tancik et al. (2020) "Fourier Features Let Networks Learn High Frequency Functions"
    """

    features: Sequence[int] = (64, 64, 64, 1)
    fourier_features: int = 32
    sigma: float = 1.0
    activation: callable = nn.tanh

    def setup(self):
        # Random Fourier features matrix (fixed after initialization)
        self.B = self.param(
            'fourier_matrix',
            lambda rng, shape: jax.random.normal(rng, shape) * self.sigma,
            (2, self.fourier_features)  # 2 = spatial_dim + time_dim
        )

    @nn.compact
    def __call__(self, x, t):
        # Concatenate inputs
        inputs = jnp.concatenate([x, t], axis=-1)

        # Fourier feature encoding: [sin(2π B^T x), cos(2π B^T x)]
        projected = 2 * jnp.pi * jnp.dot(inputs, self.B)
        fourier_features = jnp.concatenate([jnp.sin(projected), jnp.cos(projected)], axis=-1)

        # Standard MLP on Fourier features
        z = fourier_features
        for feat in self.features[:-1]:
            z = nn.Dense(feat)(z)
            z = self.activation(z)

        output = nn.Dense(self.features[-1])(z)
        return output
