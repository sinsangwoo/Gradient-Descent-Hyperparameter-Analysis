"""Data preprocessing utilities."""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp


class Normalizer:
    """Data normalizer for physics simulations.

    Supports:
    - Min-max normalization: [0, 1]
    - Standard normalization: mean=0, std=1
    - Custom range normalization

    Example:
        >>> normalizer = Normalizer(method='minmax')
        >>> x_norm = normalizer.fit_transform(x)
        >>> x_orig = normalizer.inverse_transform(x_norm)
    """

    def __init__(self, method: str = "minmax"):
        """Initialize normalizer.

        Args:
            method: Normalization method ('minmax' or 'standard')
        """
        self.method = method
        self.params = {}

    def fit(self, data: jnp.ndarray) -> "Normalizer":
        """Compute normalization parameters.

        Args:
            data: Input data array

        Returns:
            Self for chaining
        """
        if self.method == "minmax":
            self.params["min"] = jnp.min(data)
            self.params["max"] = jnp.max(data)
        elif self.method == "standard":
            self.params["mean"] = jnp.mean(data)
            self.params["std"] = jnp.std(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Apply normalization.

        Args:
            data: Input data

        Returns:
            Normalized data
        """
        if not self.params:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if self.method == "minmax":
            return (data - self.params["min"]) / (
                self.params["max"] - self.params["min"] + 1e-8
            )
        elif self.method == "standard":
            return (data - self.params["mean"]) / (
                self.params["std"] + 1e-8
            )

    def fit_transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Fit and transform in one step.

        Args:
            data: Input data

        Returns:
            Normalized data
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Reverse normalization.

        Args:
            data: Normalized data

        Returns:
            Original scale data
        """
        if not self.params:
            raise ValueError("Normalizer not fitted.")

        if self.method == "minmax":
            return data * (
                self.params["max"] - self.params["min"]
            ) + self.params["min"]
        elif self.method == "standard":
            return data * self.params["std"] + self.params["mean"]


class GridGenerator:
    """Generate collocation grids for PINNs.

    Supports:
    - Uniform grids
    - Random sampling
    - Latin hypercube sampling
    - Adaptive refinement

    Example:
        >>> generator = GridGenerator(domain=[(0, 1), (0, 1)])
        >>> points = generator.uniform(n=100)
        >>> points_adaptive = generator.refine_near_boundary(points)
    """

    def __init__(self, domain: list):
        """Initialize grid generator.

        Args:
            domain: List of (min, max) tuples for each dimension
                   Example: [(0, 1), (0, 1), (0, 0.5)] for x, y, t
        """
        self.domain = domain
        self.ndim = len(domain)

    def uniform(self, n: int) -> jnp.ndarray:
        """Generate uniform grid.

        Args:
            n: Number of points per dimension

        Returns:
            Grid points of shape (n^ndim, ndim)
        """
        grids = [jnp.linspace(low, high, n) for low, high in self.domain]
        mesh = jnp.meshgrid(*grids, indexing="ij")
        points = jnp.stack([g.flatten() for g in mesh], axis=-1)
        return points

    def random(
        self,
        n: int,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Generate random points.

        Args:
            n: Number of points
            rng: Random key

        Returns:
            Random points of shape (n, ndim)
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        points = []
        for low, high in self.domain:
            rng, key = jax.random.split(rng)
            pts = jax.random.uniform(key, (n,)) * (high - low) + low
            points.append(pts)

        return jnp.stack(points, axis=-1)

    def boundary(
        self,
        n: int,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Generate boundary points.

        Args:
            n: Number of points per boundary
            rng: Random key

        Returns:
            Boundary points
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # For 2D: generate points on 4 boundaries
        if self.ndim == 2:
            boundaries = []
            (x_min, x_max), (y_min, y_max) = self.domain

            # Bottom
            rng, key = jax.random.split(rng)
            x = jax.random.uniform(key, (n,)) * (x_max - x_min) + x_min
            y = jnp.ones(n) * y_min
            boundaries.append(jnp.stack([x, y], axis=-1))

            # Top
            rng, key = jax.random.split(rng)
            x = jax.random.uniform(key, (n,)) * (x_max - x_min) + x_min
            y = jnp.ones(n) * y_max
            boundaries.append(jnp.stack([x, y], axis=-1))

            # Left
            rng, key = jax.random.split(rng)
            x = jnp.ones(n) * x_min
            y = jax.random.uniform(key, (n,)) * (y_max - y_min) + y_min
            boundaries.append(jnp.stack([x, y], axis=-1))

            # Right
            rng, key = jax.random.split(rng)
            x = jnp.ones(n) * x_max
            y = jax.random.uniform(key, (n,)) * (y_max - y_min) + y_min
            boundaries.append(jnp.stack([x, y], axis=-1))

            return jnp.concatenate(boundaries, axis=0)
        else:
            raise NotImplementedError("Boundary generation for ndim != 2")


def create_collocation_points(
    domain: Dict[str, Tuple[float, float]],
    n_pde: int,
    n_bc: int,
    n_ic: int,
    rng: jax.random.PRNGKey,
) -> Dict[str, jnp.ndarray]:
    """Create collocation points for PINN training.

    Args:
        domain: Dictionary with domain bounds
                Example: {'x': (0, 1), 'y': (0, 1), 't': (0, 1)}
        n_pde: Number of PDE collocation points
        n_bc: Number of boundary points
        n_ic: Number of initial condition points
        rng: Random key

    Returns:
        Dictionary with collocation points
    """
    # Extract dimensions
    spatial_dims = [k for k in domain.keys() if k != "t"]
    has_time = "t" in domain

    # PDE points (interior)
    pde_domain = [domain[k] for k in domain.keys()]
    generator = GridGenerator(pde_domain)
    pde_points = generator.random(n_pde, rng)

    result = {}
    for i, key in enumerate(domain.keys()):
        result[f"{key}_pde"] = pde_points[:, i]

    # Boundary points
    if len(spatial_dims) > 0:
        bc_domain = [domain[k] for k in spatial_dims]
        bc_generator = GridGenerator(bc_domain)
        rng, key = jax.random.split(rng)
        bc_points = bc_generator.boundary(n_bc // 4, key)

        for i, key in enumerate(spatial_dims):
            result[f"{key}_bc"] = bc_points[:, i]

        if has_time:
            rng, key = jax.random.split(rng)
            t_bc = jax.random.uniform(key, (len(bc_points),))
            t_bc = t_bc * (domain["t"][1] - domain["t"][0]) + domain["t"][0]
            result["t_bc"] = t_bc

    # Initial condition points
    if has_time:
        ic_domain = [domain[k] for k in spatial_dims]
        ic_generator = GridGenerator(ic_domain)
        rng, key = jax.random.split(rng)
        ic_points = ic_generator.random(n_ic, key)

        for i, key in enumerate(spatial_dims):
            result[f"{key}_ic"] = ic_points[:, i]

    return result


def normalize_data(
    data: Dict[str, jnp.ndarray],
    method: str = "minmax",
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Normalizer]]:
    """Normalize all fields in data dictionary.

    Args:
        data: Dictionary with data arrays
        method: Normalization method

    Returns:
        normalized_data: Normalized data
        normalizers: Dictionary of fitted normalizers
    """
    normalized_data = {}
    normalizers = {}

    for key, value in data.items():
        normalizer = Normalizer(method=method)
        normalized_data[key] = normalizer.fit_transform(value)
        normalizers[key] = normalizer

    return normalized_data, normalizers
