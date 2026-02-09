"""Data loading utilities for various formats."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import jax.numpy as jnp
import numpy as np


class DataLoader:
    """Universal data loader for physics simulations.

    Supports:
    - CSV files
    - HDF5 files
    - NumPy arrays
    - JSON metadata

    Example:
        >>> loader = DataLoader()
        >>> data = loader.load('simulation_data.csv')
        >>> print(data.keys())
        ['x', 'y', 't', 'u', 'v', 'p']
    """

    SUPPORTED_FORMATS = [".csv", ".h5", ".hdf5", ".npy", ".npz"]

    def load(
        self,
        filepath: Union[str, Path],
        format: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, jnp.ndarray]:
        """Load data from file.

        Args:
            filepath: Path to data file
            format: File format (auto-detected if None)
            **kwargs: Format-specific arguments

        Returns:
            Dictionary with loaded data as JAX arrays

        Raises:
            ValueError: If format not supported
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Auto-detect format
        if format is None:
            format = filepath.suffix.lower()

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        # Load based on format
        if format == ".csv":
            return load_csv(filepath, **kwargs)
        elif format in [".h5", ".hdf5"]:
            return load_hdf5(filepath, **kwargs)
        elif format in [".npy", ".npz"]:
            return load_numpy(filepath, **kwargs)
        else:
            raise ValueError(f"Format {format} not implemented")

    def save(
        self,
        data: Dict[str, jnp.ndarray],
        filepath: Union[str, Path],
        format: Optional[str] = None,
        **kwargs,
    ):
        """Save data to file.

        Args:
            data: Dictionary with data arrays
            filepath: Path to save file
            format: File format (auto-detected if None)
            **kwargs: Format-specific arguments
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = filepath.suffix.lower()

        if format == ".npz":
            np.savez(filepath, **{k: np.array(v) for k, v in data.items()})
        elif format == ".json":
            # Save metadata only
            metadata = {k: {"shape": v.shape, "dtype": str(v.dtype)}
                       for k, v in data.items()}
            with open(filepath, "w") as f:
                json.dump(metadata, f, indent=2)
        else:
            raise ValueError(f"Save format {format} not implemented")


def load_csv(
    filepath: Union[str, Path],
    delimiter: str = ",",
    **kwargs,
) -> Dict[str, jnp.ndarray]:
    """Load data from CSV file.

    Expected format:
    - First row: column names
    - Remaining rows: data

    Args:
        filepath: Path to CSV file
        delimiter: Column delimiter
        **kwargs: Additional arguments for np.genfromtxt

    Returns:
        Dictionary mapping column names to arrays
    """
    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        names=True,
        **kwargs,
    )

    return {name: jnp.array(data[name]) for name in data.dtype.names}


def load_hdf5(
    filepath: Union[str, Path],
    group: str = "/",
    **kwargs,
) -> Dict[str, jnp.ndarray]:
    """Load data from HDF5 file.

    Args:
        filepath: Path to HDF5 file
        group: HDF5 group to load from
        **kwargs: Additional arguments

    Returns:
        Dictionary with datasets from HDF5 file
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py required for HDF5 support. Install with: pip install h5py"
        )

    result = {}
    with h5py.File(filepath, "r") as f:
        grp = f[group]
        for key in grp.keys():
            result[key] = jnp.array(grp[key][:])

    return result


def load_numpy(
    filepath: Union[str, Path],
    **kwargs,
) -> Dict[str, jnp.ndarray]:
    """Load data from NumPy file (.npy or .npz).

    Args:
        filepath: Path to NumPy file
        **kwargs: Additional arguments

    Returns:
        Dictionary with loaded arrays
    """
    filepath = Path(filepath)

    if filepath.suffix == ".npy":
        data = np.load(filepath, **kwargs)
        return {"data": jnp.array(data)}
    elif filepath.suffix == ".npz":
        data = np.load(filepath, **kwargs)
        return {key: jnp.array(data[key]) for key in data.files}
    else:
        raise ValueError(f"Unknown NumPy format: {filepath.suffix}")
