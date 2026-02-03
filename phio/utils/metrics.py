"""Evaluation metrics for PINN solutions."""

from typing import Dict
import jax.numpy as jnp


def compute_l2_error(
    u_pred: jnp.ndarray, u_exact: jnp.ndarray, relative: bool = True
) -> float:
    """Compute L2 error between predicted and exact solutions.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution
        relative: If True, compute relative error; otherwise absolute

    Returns:
        L2 error (scalar)
    """
    numerator = jnp.linalg.norm(u_pred - u_exact)
    if relative:
        denominator = jnp.linalg.norm(u_exact)
        return float(numerator / (denominator + 1e-10))
    else:
        return float(numerator)


def compute_max_error(u_pred: jnp.ndarray, u_exact: jnp.ndarray) -> float:
    """Compute maximum absolute error.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution

    Returns:
        Maximum error (scalar)
    """
    return float(jnp.max(jnp.abs(u_pred - u_exact)))


def compute_metrics(u_pred: jnp.ndarray, u_exact: jnp.ndarray) -> Dict[str, float]:
    """Compute comprehensive error metrics.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution

    Returns:
        Dictionary with various metrics
    """
    l2_rel = compute_l2_error(u_pred, u_exact, relative=True)
    l2_abs = compute_l2_error(u_pred, u_exact, relative=False)
    max_err = compute_max_error(u_pred, u_exact)
    mean_err = float(jnp.mean(jnp.abs(u_pred - u_exact)))

    return {
        "l2_relative": l2_rel,
        "l2_absolute": l2_abs,
        "max_error": max_err,
        "mean_error": mean_err,
    }
