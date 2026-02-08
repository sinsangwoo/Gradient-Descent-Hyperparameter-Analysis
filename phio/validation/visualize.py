"""Visualization tools for validation results."""

from typing import Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_benchmark_comparison(
    predictions: Dict[str, jnp.ndarray],
    reynolds_number: int,
    save_path: str = "benchmark_comparison.png",
):
    """Plot PINN predictions vs Ghia benchmark data.

    Args:
        predictions: Dictionary from GhiaCavityData.compare_with_pinn
        reynolds_number: Reynolds number
        save_path: Output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # U-velocity comparison
    axes[0].plot(
        predictions["u_benchmark"],
        predictions["y_coords"],
        "ko",
        markersize=8,
        label="Ghia et al. (1982)",
        zorder=3,
    )
    axes[0].plot(
        predictions["u_pred"],
        predictions["y_coords"],
        "r-",
        linewidth=2,
        label="PINN",
        zorder=2,
    )
    axes[0].set_xlabel("u-velocity", fontsize=12)
    axes[0].set_ylabel("y", fontsize=12)
    axes[0].set_title(
        f"U-Velocity at x=0.5 (Re={reynolds_number})",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].legend(fontsize=11, loc="best")
    axes[0].grid(True, alpha=0.3)

    # V-velocity comparison
    axes[1].plot(
        predictions["x_coords"],
        predictions["v_benchmark"],
        "ko",
        markersize=8,
        label="Ghia et al. (1982)",
        zorder=3,
    )
    axes[1].plot(
        predictions["x_coords"],
        predictions["v_pred"],
        "b-",
        linewidth=2,
        label="PINN",
        zorder=2,
    )
    axes[1].set_xlabel("x", fontsize=12)
    axes[1].set_ylabel("v-velocity", fontsize=12)
    axes[1].set_title(
        f"V-Velocity at y=0.5 (Re={reynolds_number})",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].legend(fontsize=11, loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nBenchmark comparison saved to {save_path}")


def plot_error_distribution(
    predictions: Dict[str, jnp.ndarray],
    save_path: str = "error_distribution.png",
):
    """Plot error distribution and residuals.

    Args:
        predictions: Dictionary from GhiaCavityData.compare_with_pinn
        save_path: Output file path
    """
    u_error = predictions["u_pred"] - predictions["u_benchmark"]
    v_error = predictions["v_pred"] - predictions["v_benchmark"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # U-velocity error vs position
    axes[0, 0].plot(
        predictions["y_coords"], u_error, "ro-", linewidth=2, markersize=6
    )
    axes[0, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("y", fontsize=11)
    axes[0, 0].set_ylabel("Error in u", fontsize=11)
    axes[0, 0].set_title("U-Velocity Error", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    # V-velocity error vs position
    axes[0, 1].plot(
        predictions["x_coords"], v_error, "bo-", linewidth=2, markersize=6
    )
    axes[0, 1].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("x", fontsize=11)
    axes[0, 1].set_ylabel("Error in v", fontsize=11)
    axes[0, 1].set_title("V-Velocity Error", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # Error histogram for u
    axes[1, 0].hist(u_error, bins=15, color="red", alpha=0.7, edgecolor="black")
    axes[1, 0].axvline(0, color="k", linestyle="--", linewidth=2)
    axes[1, 0].set_xlabel("Error in u", fontsize=11)
    axes[1, 0].set_ylabel("Frequency", fontsize=11)
    axes[1, 0].set_title("U-Velocity Error Distribution", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Error histogram for v
    axes[1, 1].hist(v_error, bins=15, color="blue", alpha=0.7, edgecolor="black")
    axes[1, 1].axvline(0, color="k", linestyle="--", linewidth=2)
    axes[1, 1].set_xlabel("Error in v", fontsize=11)
    axes[1, 1].set_ylabel("Frequency", fontsize=11)
    axes[1, 1].set_title("V-Velocity Error Distribution", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Error distribution saved to {save_path}")


def plot_validation_dashboard(
    predictions: Dict[str, jnp.ndarray],
    u_metrics: Dict[str, float],
    v_metrics: Dict[str, float],
    reynolds_number: int,
    save_path: str = "validation_dashboard.png",
):
    """Create comprehensive validation dashboard.

    Args:
        predictions: Dictionary from GhiaCavityData.compare_with_pinn
        u_metrics: Error metrics for u-velocity
        v_metrics: Error metrics for v-velocity
        reynolds_number: Reynolds number
        save_path: Output file path
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(
        f"PINN Validation Dashboard (Re = {reynolds_number})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # 1. U-velocity comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        predictions["u_benchmark"],
        predictions["y_coords"],
        "ko",
        markersize=6,
        label="Benchmark",
    )
    ax1.plot(
        predictions["u_pred"], predictions["y_coords"], "r-", linewidth=2, label="PINN"
    )
    ax1.set_xlabel("u")
    ax1.set_ylabel("y")
    ax1.set_title("U-Velocity (x=0.5)", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. V-velocity comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        predictions["x_coords"],
        predictions["v_benchmark"],
        "ko",
        markersize=6,
        label="Benchmark",
    )
    ax2.plot(
        predictions["x_coords"], predictions["v_pred"], "b-", linewidth=2, label="PINN"
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("v")
    ax2.set_title("V-Velocity (y=0.5)", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Error metrics table
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    table_data = [
        ["Metric", "U-Velocity", "V-Velocity"],
        ["MAE", f"{u_metrics['mae']:.4f}", f"{v_metrics['mae']:.4f}"],
        ["RMSE", f"{u_metrics['rmse']:.4f}", f"{v_metrics['rmse']:.4f}"],
        ["Max Error", f"{u_metrics['max_error']:.4f}", f"{v_metrics['max_error']:.4f}"],
        [
            "Relative L2",
            f"{u_metrics['relative_l2']:.4f}",
            f"{v_metrics['relative_l2']:.4f}",
        ],
    ]
    table = ax3.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.35, 0.325, 0.325],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")
    ax3.set_title("Error Metrics", fontweight="bold", pad=20)

    # 4. U-velocity error
    ax4 = fig.add_subplot(gs[1, 0])
    u_error = predictions["u_pred"] - predictions["u_benchmark"]
    ax4.plot(predictions["y_coords"], u_error, "ro-", linewidth=2, markersize=5)
    ax4.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax4.set_xlabel("y")
    ax4.set_ylabel("Error")
    ax4.set_title("U-Velocity Error", fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # 5. V-velocity error
    ax5 = fig.add_subplot(gs[1, 1])
    v_error = predictions["v_pred"] - predictions["v_benchmark"]
    ax5.plot(predictions["x_coords"], v_error, "bo-", linewidth=2, markersize=5)
    ax5.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax5.set_xlabel("x")
    ax5.set_ylabel("Error")
    ax5.set_title("V-Velocity Error", fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # 6. Combined error histogram
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(u_error, bins=10, alpha=0.7, label="U-error", color="red", edgecolor="black")
    ax6.hist(v_error, bins=10, alpha=0.7, label="V-error", color="blue", edgecolor="black")
    ax6.axvline(0, color="k", linestyle="--", linewidth=2)
    ax6.set_xlabel("Error")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Error Distribution", fontweight="bold")
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")

    # 7-9. Scatter plots (predicted vs actual)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(predictions["u_benchmark"], predictions["u_pred"], alpha=0.6, s=50)
    lim = [predictions["u_benchmark"].min(), predictions["u_benchmark"].max()]
    ax7.plot(lim, lim, "k--", alpha=0.5, linewidth=2)
    ax7.set_xlabel("Benchmark U")
    ax7.set_ylabel("PINN U")
    ax7.set_title("U-Velocity: Predicted vs Actual", fontweight="bold")
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(predictions["v_benchmark"], predictions["v_pred"], alpha=0.6, s=50, color="blue")
    lim = [predictions["v_benchmark"].min(), predictions["v_benchmark"].max()]
    ax8.plot(lim, lim, "k--", alpha=0.5, linewidth=2)
    ax8.set_xlabel("Benchmark V")
    ax8.set_ylabel("PINN V")
    ax8.set_title("V-Velocity: Predicted vs Actual", fontweight="bold")
    ax8.grid(True, alpha=0.3)

    # 9. Quality assessment
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    avg_rel_l2 = (u_metrics["relative_l2"] + v_metrics["relative_l2"]) / 2

    if avg_rel_l2 < 0.01:
        quality = "EXCELLENT"
        color = "green"
    elif avg_rel_l2 < 0.05:
        quality = "GOOD"
        color = "blue"
    elif avg_rel_l2 < 0.10:
        quality = "ACCEPTABLE"
        color = "orange"
    else:
        quality = "NEEDS IMPROVEMENT"
        color = "red"

    ax9.text(
        0.5,
        0.6,
        quality,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=color,
    )
    ax9.text(
        0.5,
        0.4,
        f"Avg Relative L2: {avg_rel_l2:.4f}",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax9.set_title("Overall Quality", fontweight="bold", pad=20)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Validation dashboard saved to {save_path}")
