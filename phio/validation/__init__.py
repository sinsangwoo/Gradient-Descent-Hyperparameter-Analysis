"""Validation tools for comparing PINN results with benchmarks."""

from phio.validation.metrics import compute_error_metrics, generate_error_report
from phio.validation.visualize import (
    plot_benchmark_comparison,
    plot_error_distribution,
    plot_validation_dashboard,
)

__all__ = [
    "compute_error_metrics",
    "generate_error_report",
    "plot_benchmark_comparison",
    "plot_error_distribution",
    "plot_validation_dashboard",
]
