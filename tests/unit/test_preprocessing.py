"""Unit tests for data preprocessing."""

import jax
import jax.numpy as jnp
import pytest

from phio.data import (
    GridGenerator,
    Normalizer,
    create_collocation_points,
    normalize_data,
)


class TestNormalizer:
    """Test Normalizer class."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        normalizer = Normalizer(method="minmax")
        normalized = normalizer.fit_transform(data)

        # Should be in [0, 1]
        assert jnp.min(normalized) >= 0.0
        assert jnp.max(normalized) <= 1.0
        assert jnp.isclose(jnp.min(normalized), 0.0)
        assert jnp.isclose(jnp.max(normalized), 1.0)

    def test_standard_normalization(self):
        """Test standard normalization."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        normalizer = Normalizer(method="standard")
        normalized = normalizer.fit_transform(data)

        # Should have mean~0, std~1
        assert jnp.abs(jnp.mean(normalized)) < 1e-6
        assert jnp.abs(jnp.std(normalized) - 1.0) < 1e-6

    def test_inverse_transform(self):
        """Test inverse transformation."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        normalizer = Normalizer(method="minmax")
        normalized = normalizer.fit_transform(data)
        recovered = normalizer.inverse_transform(normalized)

        assert jnp.allclose(data, recovered, atol=1e-6)

    def test_not_fitted_error(self):
        """Test error when not fitted."""
        normalizer = Normalizer()
        data = jnp.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="not fitted"):
            normalizer.transform(data)


class TestGridGenerator:
    """Test GridGenerator class."""

    def test_uniform_grid(self):
        """Test uniform grid generation."""
        domain = [(0.0, 1.0), (0.0, 1.0)]
        generator = GridGenerator(domain)

        points = generator.uniform(n=10)

        # Check shape
        assert points.shape == (100, 2)

        # Check bounds
        assert jnp.min(points[:, 0]) >= 0.0
        assert jnp.max(points[:, 0]) <= 1.0

    def test_random_grid(self):
        """Test random point generation."""
        domain = [(0.0, 1.0), (0.0, 1.0)]
        generator = GridGenerator(domain)

        rng = jax.random.PRNGKey(42)
        points = generator.random(n=50, rng=rng)

        # Check shape
        assert points.shape == (50, 2)

        # Check bounds
        assert jnp.all(points >= 0.0)
        assert jnp.all(points <= 1.0)

    def test_boundary_generation_2d(self):
        """Test boundary point generation for 2D."""
        domain = [(0.0, 1.0), (0.0, 1.0)]
        generator = GridGenerator(domain)

        rng = jax.random.PRNGKey(42)
        points = generator.boundary(n=10, rng=rng)

        # Should have 4*10 = 40 points (4 boundaries)
        assert points.shape == (40, 2)


class TestCollocationPoints:
    """Test collocation point generation."""

    def test_create_collocation_points(self):
        """Test collocation point creation."""
        domain = {"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 1.0)}
        rng = jax.random.PRNGKey(42)

        points = create_collocation_points(
            domain=domain,
            n_pde=100,
            n_bc=20,
            n_ic=20,
            rng=rng,
        )

        # Check PDE points
        assert "x_pde" in points
        assert "y_pde" in points
        assert "t_pde" in points
        assert len(points["x_pde"]) == 100

        # Check boundary points
        assert "x_bc" in points
        assert "y_bc" in points
        assert "t_bc" in points

        # Check IC points
        assert "x_ic" in points
        assert "y_ic" in points


class TestNormalizeData:
    """Test batch data normalization."""

    def test_normalize_dict(self):
        """Test normalizing data dictionary."""
        data = {
            "x": jnp.array([1.0, 2.0, 3.0]),
            "y": jnp.array([10.0, 20.0, 30.0]),
        }

        normalized, normalizers = normalize_data(data)

        # Check keys
        assert "x" in normalized
        assert "y" in normalized
        assert "x" in normalizers
        assert "y" in normalizers

        # Check normalization
        assert jnp.min(normalized["x"]) >= 0.0
        assert jnp.max(normalized["x"]) <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
