"""Visualization utilities for PINN solutions."""

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_solution(
    x: jnp.ndarray,
    t: jnp.ndarray,
    u_pred: jnp.ndarray,
    u_exact: Optional[jnp.ndarray] = None,
    title: str = "PINN Solution",
    save_path: Optional[str] = None,
):
    """Plot 1D solution over time.

    Args:
        x: Spatial coordinates (1D array)
        t: Time coordinates (1D array)
        u_pred: Predicted solution (2D array: time x space)
        u_exact: Exact solution (optional, same shape as u_pred)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2 if u_exact is not None else 1, figsize=(12, 4))
    if u_exact is None:
        axes = [axes]

    # Plot predicted solution
    im1 = axes[0].contourf(x, t, u_pred, levels=50, cmap="viridis")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title(f"{title} (Predicted)")
    plt.colorbar(im1, ax=axes[0])

    # Plot exact solution (if provided)
    if u_exact is not None:
        im2 = axes[1].contourf(x, t, u_exact, levels=50, cmap="viridis")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("t")
        axes[1].set_title(f"{title} (Exact)")
        plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_loss_history(
    loss_history: List[float],
    log_scale: bool = True,
    title: str = "Training Loss",
    save_path: Optional[str] = None,
):
    """Plot training loss history.

    Args:
        loss_history: List of loss values over epochs
        log_scale: Use log scale for y-axis
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    if log_scale:
        plt.yscale("log")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_error_distribution(
    x: jnp.ndarray,
    t: jnp.ndarray,
    error: jnp.ndarray,
    title: str = "Absolute Error",
    save_path: Optional[str] = None,
):
    """Plot error distribution.

    Args:
        x: Spatial coordinates
        t: Time coordinates
        error: Absolute error |u_pred - u_exact|
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    im = plt.contourf(x, t, error, levels=50, cmap="hot")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("t", fontsize=12)
    plt.title(title, fontsize=14)
    plt.colorbar(im, label="Absolute Error")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
