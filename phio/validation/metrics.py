"""Quantitative validation metrics for PINN results."""

from typing import Dict

import jax.numpy as jnp


def compute_error_metrics(
    predictions: jnp.ndarray,
    ground_truth: jnp.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive error metrics.

    Args:
        predictions: Model predictions
        ground_truth: Reference values

    Returns:
        Dictionary with error metrics:
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - max_error: Maximum absolute error
            - relative_l2: Relative L2 error
    """
    error = predictions - ground_truth

    mae = float(jnp.mean(jnp.abs(error)))
    mse = float(jnp.mean(error**2))
    rmse = float(jnp.sqrt(mse))
    max_error = float(jnp.max(jnp.abs(error)))

    # Relative L2 error: ||pred - true||_2 / ||true||_2
    relative_l2 = float(
        jnp.linalg.norm(error) / (jnp.linalg.norm(ground_truth) + 1e-10)
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_error": max_error,
        "relative_l2": relative_l2,
    }


def generate_error_report(
    u_metrics: Dict[str, float],
    v_metrics: Dict[str, float],
    reynolds_number: int,
) -> str:
    """Generate formatted error report.

    Args:
        u_metrics: Error metrics for u-velocity
        v_metrics: Error metrics for v-velocity
        reynolds_number: Reynolds number

    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 60)
    report.append(f"VALIDATION REPORT: Re = {reynolds_number}")
    report.append("=" * 60)
    report.append("")

    report.append("U-Velocity (Vertical Centerline):")
    report.append(f"  MAE:          {u_metrics['mae']:.6f}")
    report.append(f"  RMSE:         {u_metrics['rmse']:.6f}")
    report.append(f"  Max Error:    {u_metrics['max_error']:.6f}")
    report.append(f"  Relative L2:  {u_metrics['relative_l2']:.6f}")
    report.append("")

    report.append("V-Velocity (Horizontal Centerline):")
    report.append(f"  MAE:          {v_metrics['mae']:.6f}")
    report.append(f"  RMSE:         {v_metrics['rmse']:.6f}")
    report.append(f"  Max Error:    {v_metrics['max_error']:.6f}")
    report.append(f"  Relative L2:  {v_metrics['relative_l2']:.6f}")
    report.append("")

    # Overall assessment
    avg_mae = (u_metrics["mae"] + v_metrics["mae"]) / 2
    avg_rel_l2 = (u_metrics["relative_l2"] + v_metrics["relative_l2"]) / 2

    report.append("Overall Assessment:")
    report.append(f"  Average MAE:        {avg_mae:.6f}")
    report.append(f"  Average Relative L2: {avg_rel_l2:.6f}")

    if avg_rel_l2 < 0.01:
        assessment = "EXCELLENT (<1% error)"
    elif avg_rel_l2 < 0.05:
        assessment = "GOOD (1-5% error)"
    elif avg_rel_l2 < 0.10:
        assessment = "ACCEPTABLE (5-10% error)"
    else:
        assessment = "NEEDS IMPROVEMENT (>10% error)"

    report.append(f"  Quality:            {assessment}")
    report.append("=" * 60)

    return "\n".join(report)
