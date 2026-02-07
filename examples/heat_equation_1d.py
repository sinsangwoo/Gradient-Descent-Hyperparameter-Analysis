#!/usr/bin/env python3
"""Example: Solve 1D Heat Equation with Physics-Informed Neural Network.

This script demonstrates PhIO's core capabilities:
1. Define PDE (heat equation) with analytical solution
2. Train PINN with curriculum learning
3. Visualize results and compare with ground truth
4. Benchmark accuracy (L2 error) and speed

Usage:
    python examples/heat_equation_1d.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from phio.networks.mlp import FourierFeatureMLP
from phio.physics.heat import (
    analytical_gaussian,
    heat_equation_residual,
)
from phio.solvers.pinn_trainer import (
    create_train_state,
    train_pinn,
)

# Seed for reproducibility
jax.config.update("jax_enable_x64", True)  # Use float64 for better accuracy
rng = jax.random.PRNGKey(42)


def generate_training_data(
    n_pde: int = 10000,
    n_bc: int = 200,
    n_ic: int = 200,
    x_range: tuple = (0.0, 1.0),
    t_range: tuple = (0.0, 1.0),
    alpha: float = 0.01,
):
    """Generate collocation points for PINN training.

    Args:
        n_pde: Number of interior PDE collocation points
        n_bc: Number of boundary condition points
        n_ic: Number of initial condition points
        x_range: Spatial domain
        t_range: Time domain
        alpha: Thermal diffusivity

    Returns:
        Dictionary with training data
    """
    rng_local = jax.random.PRNGKey(123)

    # PDE collocation points (Latin Hypercube Sampling for better coverage)
    rng_local, rng_x, rng_t = jax.random.split(rng_local, 3)
    x_pde = jax.random.uniform(rng_x, (n_pde, 1), minval=x_range[0], maxval=x_range[1])
    t_pde = jax.random.uniform(rng_t, (n_pde, 1), minval=t_range[0], maxval=t_range[1])

    # Boundary conditions: u(0, t) = u(1, t) = 0 (Dirichlet)
    t_bc_vals = np.linspace(t_range[0], t_range[1], n_bc // 2)
    x_bc_left = np.zeros(n_bc // 2)
    x_bc_right = np.ones(n_bc // 2)
    x_bc = jnp.array(np.concatenate([x_bc_left, x_bc_right]))
    t_bc = jnp.array(np.concatenate([t_bc_vals, t_bc_vals]))
    u_bc = jnp.zeros_like(x_bc)  # Zero at boundaries

    # Initial condition: Gaussian pulse
    x_ic_vals = np.linspace(x_range[0], x_range[1], n_ic)
    x_ic = jnp.array(x_ic_vals)
    u_ic = analytical_gaussian(
        x_ic, jnp.zeros_like(x_ic), alpha=alpha, x0=0.5, sigma0=0.1
    )
    u_ic = u_ic.squeeze()

    return {
        "x_pde": x_pde,
        "t_pde": t_pde,
        "x_bc": x_bc,
        "t_bc": t_bc,
        "u_bc": u_bc,
        "x_ic": x_ic,
        "u_ic": u_ic,
    }


def main():
    """Main training and evaluation pipeline."""
    print("=" * 70)
    print("PhIO: 1D Heat Equation PINN Solver")
    print("=" * 70)

    # Problem parameters
    alpha = 0.01  # Thermal diffusivity
    num_epochs = 10000
    learning_rate = 1e-3

    # Generate training data
    print("\n[1/5] Generating training data...")
    data = generate_training_data(n_pde=10000, n_bc=200, n_ic=200, alpha=alpha)
    print(f"  - PDE points: {data['x_pde'].shape[0]}")
    print(f"  - BC points: {data['x_bc'].shape[0]}")
    print(f"  - IC points: {data['x_ic'].shape[0]}")

    # Initialize model
    print("\n[2/5] Initializing PINN (Fourier Feature MLP)...")
    model = FourierFeatureMLP(
        features=[128, 128, 128, 1], fourier_features=64, sigma=5.0
    )

    state = create_train_state(
        rng,
        model,
        learning_rate=learning_rate,
        sample_input=(data["x_pde"][:1], data["t_pde"][:1]),
    )
    print(f"  - Network: {model}")
    print(
        f"  - Parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params))}"
    )

    # Curriculum learning schedule
    print("\n[3/5] Training with curriculum learning...")
    curriculum = {
        0: {"ic": 100.0, "bc": 10.0, "pde": 1.0},  # Phase 1: Learn IC first
        2000: {"ic": 10.0, "bc": 10.0, "pde": 1.0},  # Phase 2: Balance IC/BC
        5000: {"ic": 1.0, "bc": 1.0, "pde": 1.0},  # Phase 3: Equal weights
    }

    start_time = time.time()
    state, history = train_pinn(
        state,
        data["x_pde"],
        data["t_pde"],
        data["x_bc"],
        data["t_bc"],
        data["u_bc"],
        data["x_ic"],
        data["u_ic"],
        pde_residual_fn=heat_equation_residual,
        alpha=alpha,
        num_epochs=num_epochs,
        print_every=2000,
        curriculum_schedule=curriculum,
    )
    train_time = time.time() - start_time
    print(
        f"\n  Training completed in {train_time:.2f}s "
        f"({train_time/num_epochs*1000:.2f}ms/epoch)"
    )

    # Evaluate on test grid
    print("\n[4/5] Evaluating on test grid...")
    nx, nt = 100, 50
    x_test = jnp.linspace(0, 1, nx)
    t_test = jnp.linspace(0, 1, nt)
    X_test, T_test = jnp.meshgrid(x_test, t_test)

    # PINN prediction
    u_pred = jax.vmap(
        jax.vmap(state.apply_fn, in_axes=(None, 0, None)), in_axes=(None, None, 0)
    )(state.params, X_test, T_test).squeeze()

    # Analytical solution
    u_true = analytical_gaussian(x_test, t_test, alpha=alpha, x0=0.5, sigma0=0.1)

    # Compute error
    l2_error = jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true)
    max_error = jnp.max(jnp.abs(u_pred - u_true))

    print(f"  - L2 relative error: {l2_error:.6f}")
    print(f"  - Max absolute error: {max_error:.6f}")

    # Visualize results
    print("\n[5/5] Generating visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("PhIO: 1D Heat Equation Results", fontsize=16, fontweight="bold")

    # Loss curves
    ax = axes[0, 0]
    ax.semilogy(history["total"], label="Total Loss", linewidth=2)
    ax.semilogy(history["pde"], label="PDE Loss", alpha=0.7)
    ax.semilogy(history["bc"], label="BC Loss", alpha=0.7)
    ax.semilogy(history["ic"], label="IC Loss", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PINN solution
    ax = axes[0, 1]
    im1 = ax.contourf(X_test, T_test, u_pred, levels=50, cmap="hot")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"PINN Solution (L2 error: {l2_error:.4f})")
    plt.colorbar(im1, ax=ax)

    # Analytical solution
    ax = axes[0, 2]
    im2 = ax.contourf(X_test, T_test, u_true, levels=50, cmap="hot")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Analytical Solution")
    plt.colorbar(im2, ax=ax)

    # Error distribution
    ax = axes[1, 0]
    error = jnp.abs(u_pred - u_true)
    im3 = ax.contourf(X_test, T_test, error, levels=50, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Absolute Error")
    plt.colorbar(im3, ax=ax)

    # Snapshots at different times
    ax = axes[1, 1]
    for t_snap in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = int(t_snap * (nt - 1))
        ax.plot(x_test, u_pred[idx, :], label=f"PINN (t={t_snap:.2f})", linestyle="-")
        ax.plot(
            x_test,
            u_true[idx, :],
            label=f"True (t={t_snap:.2f})",
            linestyle="--",
            alpha=0.6,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("Temperature Profiles")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Performance summary
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = f"""
    Performance Summary
    {'='*40}

    Training:
      • Epochs: {num_epochs:,}
      • Time: {train_time:.2f}s
      • Speed: {train_time/num_epochs*1000:.2f}ms/epoch

    Accuracy:
      • L2 Error: {l2_error:.6f}
      • Max Error: {max_error:.6f}

    Model:
      • Type: Fourier MLP
      • Parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}

    Physics:
      • PDE: ∂u/∂t = α ∂²u/∂x²
      • α: {alpha}
      • Domain: x∈[0,1], t∈[0,1]
    """
    ax.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )

    plt.tight_layout()

    # Save figure
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "heat_equation_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Results saved to: {output_path}")

    plt.show()

    print("\n" + "=" * 70)
    print("SUCCESS! PhIO solved 1D heat equation with high accuracy.")
    print("=" * 70)


if __name__ == "__main__":
    main()
