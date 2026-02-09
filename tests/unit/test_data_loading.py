"""Unit tests for data loading utilities."""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phio.data import DataLoader, load_csv, load_numpy


class TestDataLoader:
    """Test DataLoader class."""

    def test_load_numpy_npy(self, tmp_path):
        """Test loading .npy files."""
        # Create test file
        test_data = np.array([1.0, 2.0, 3.0])
        filepath = tmp_path / "test.npy"
        np.save(filepath, test_data)

        # Load
        loader = DataLoader()
        result = loader.load(filepath)

        assert "data" in result
        assert jnp.allclose(result["data"], test_data)

    def test_load_numpy_npz(self, tmp_path):
        """Test loading .npz files."""
        # Create test file
        test_data = {"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6])}
        filepath = tmp_path / "test.npz"
        np.savez(filepath, **test_data)

        # Load
        loader = DataLoader()
        result = loader.load(filepath)

        assert "x" in result
        assert "y" in result
        assert jnp.allclose(result["x"], test_data["x"])

    def test_unsupported_format(self, tmp_path):
        """Test error for unsupported format."""
        filepath = tmp_path / "test.txt"
        filepath.write_text("test")

        loader = DataLoader()
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load(filepath)

    def test_file_not_found(self):
        """Test error for missing file."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.npy")

    def test_save_npz(self, tmp_path):
        """Test saving data to .npz."""
        data = {
            "x": jnp.array([1, 2, 3]),
            "y": jnp.array([4, 5, 6]),
        }
        filepath = tmp_path / "output.npz"

        loader = DataLoader()
        loader.save(data, filepath)

        # Verify file exists
        assert filepath.exists()

        # Load and verify
        loaded = loader.load(filepath)
        assert jnp.allclose(loaded["x"], data["x"])


class TestCSVLoader:
    """Test CSV loading."""

    def test_load_csv_basic(self, tmp_path):
        """Test basic CSV loading."""
        # Create test CSV
        filepath = tmp_path / "test.csv"
        content = "x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n"
        filepath.write_text(content)

        # Load
        result = load_csv(filepath)

        assert "x" in result
        assert "y" in result
        assert "z" in result
        assert len(result["x"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
