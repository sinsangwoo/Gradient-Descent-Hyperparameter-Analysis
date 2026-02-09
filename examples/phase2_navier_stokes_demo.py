"""Navier-Stokes PINN demonstration: Lid-driven cavity flow.

Benchmark problem:
- 2D square cavity [0,1] x [0,1]
- Top wall moves with velocity u_lid = 1.0
- Other walls stationary (no-slip)
- Reynolds number Re = u_lid * L / nu

Physics:
- Momentum equations (x, y directions)
- Continuity equation (incompressibility)
- Steady-state solution
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import linen as nn

from phio.physics.navier_stokes import lid_driven_cavity_bc
from phio.solvers.ns_pinn import create_ns_train_state, train_ns_pinn


class NSNetwork(nn.Module):
    """Neural network for Navier-Stokes [u, v, p]."""

    hidden_dim: int = 128
    num_layers: int = 4

    @nn.compact
    def __call__(self, x, y, t):
        inputs = jnp.concatenate([x, y, t], axis=-1)

        h = inputs
        for _ in range(self.num_layers):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.tanh(h)

        out = nn.Dense(3)(h)  # [u, v, p]
        return out


def generate_training_data(
    rng: jax.random.PRNGKey,
    n_pde: int = 2000,
    n_bc: int = 100,
) -> dict:
    """Generate collocation points for lid-driven cavity.

    Args:
        rng: Random key
        n_pde: Number of interior points
        n_bc: Number of boundary points per wall

    Returns:
        data: Dictionary with training data
    """
    # Interior collocation points
    rng, key = jax.random.split(rng)
    x_pde = jax.random.uniform(key, (n_pde,))
    y_pde = jax.random.uniform(key, (n_pde,))
    t_pde = jnp.zeros(n_pde)  # Steady-state

    # Boundary points (4 walls)
    # Top wall (y=1, moving)
    x_top = jnp.linspace(0, 1, n_bc)
    y_top = jnp.ones(n_bc)

    # Bottom wall (y=0, stationary)
    x_bottom = jnp.linspace(0, 1, n_bc)
    y_bottom = jnp.zeros(n_bc)

    # Left wall (x=0, stationary)
    x_left = jnp.zeros(n_bc)
    y_left = jnp.linspace(0, 1, n_bc)

    # Right wall (x=1, stationary)
    x_right = jnp.ones(n_bc)
    y_right = jnp.linspace(0, 1, n_bc)

    # Combine all boundaries
    x_bc = jnp.concatenate([x_top, x_bottom, x_left, x_right])
    y_bc = jnp.concatenate([y_top, y_bottom, y_left, y_right])
    t_bc = jnp.zeros_like(x_bc)

    # Boundary values
    u_bc, v_bc = lid_driven_cavity_bc(x_bc, y_bc, u_lid=1.0)

    # Initial condition (quiescent flow at t=0)
    rng, key = jax.random.split(rng)
    x_ic = jax.random.uniform(key, (100,))
    y_ic = jax.random.uniform(key, (100,))
    u_ic = jnp.zeros(100)
    v_ic = jnp.zeros(100)

    return {
        "x_pde": x_pde,
        "y_pde": y_pde,
        "t_pde": t_pde,
        "x_bc": x_bc,
        "y_bc": y_bc,
        "t_bc": t_bc,
        "u_bc": u_bc,
        "v_bc": v_bc,
        "x_ic": x_ic,
        "y_ic": y_ic,
        "u_ic": u_ic,
        "v_ic": v_ic,
    }


def visualize_solution(
    state,
    nu: float,
    save_path: str = "navier_stokes_results.png",
):
    """Visualize velocity field and streamlines.

    Args:
        state: Trained model state
        nu: Kinematic viscosity
        save_path: Output file path
    """
    # Create grid for visualization
    n_grid = 100
    x = jnp.linspace(0, 1, n_grid)
    y = jnp.linspace(0, 1, n_grid)
    X, Y = jnp.meshgrid(x, y)

    # Flatten for batch prediction
    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_flat = jnp.zeros_like(x_flat)

    # Predict velocity and pressure
    uvp = jax.vmap(state.apply_fn, in_axes=(None, 0, 0, 0))(
        state.params,
        x_flat[:, None],
        y_flat[:, None],
        t_flat[:, None],
    )

    U = uvp[:, 0].reshape(n_grid, n_grid)
    V = uvp[:, 1].reshape(n_grid, n_grid)
    P = uvp[:, 2].reshape(n_grid, n_grid)

    # Compute velocity magnitude
    speed = jnp.sqrt(U**2 + V**2)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Velocity magnitude (speed)
    im1 = axes[0, 0].contourf(X, Y, speed, levels=20, cmap="viridis")
    axes[0, 0].set_title("Velocity Magnitude |u|", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. Streamlines
    axes[0, 1].streamplot(X, Y, U, V, color=speed, cmap="viridis", linewidth=1.5, density=2)
    axes[0, 1].set_title("Streamlines", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].set_aspect("equal")

    # 3. Pressure field
    im3 = axes[1, 0].contourf(X, Y, P, levels=20, cmap="RdBu_r")
    axes[1, 0].set_title("Pressure Field p", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    axes[1, 0].set_aspect("equal")
    plt.colorbar(im3, ax=axes[1, 0])

    # 4. Velocity vectors (quiver)
    skip = 5  # Plot every 5th point
    axes[1, 1].quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        U[::skip, ::skip],
        V[::skip, ::skip],
        speed[::skip, ::skip],
        cmap="viridis",
    )
    axes[1, 1].set_title("Velocity Vectors", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].set_aspect("equal")

    # Add Reynolds number annotation
    Re = 1.0 / nu  # Re = u_lid * L / nu, with u_lid=1, L=1
    fig.suptitle(
        f"Lid-Driven Cavity Flow (Re = {Re:.1f})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nResults saved to {save_path}")


def plot_training_history(history: dict, save_path: str = "training_history.png"):
    """Plot training loss curves.

    Args:
        history: Training history dictionary
        save_path: Output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total loss
    axes[0].semilogy(history["total"], linewidth=2)
    axes[0].set_title("Total Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].grid(True, alpha=0.3)

    # Individual components
    axes[1].semilogy(history["momentum"], label="Momentum", linewidth=2)
    axes[1].semilogy(history["continuity"], label="Continuity", linewidth=2)
    axes[1].semilogy(history["bc"], label="Boundary", linewidth=2)
    axes[1].semilogy(history["ic"], label="Initial", linewidth=2)
    axes[1].set_title("Loss Components", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (log scale)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training history saved to {save_path}")


def main():
    """Main demonstration."""
    print("=" * 60)
    print("NAVIER-STOKES PINN: LID-DRIVEN CAVITY FLOW")
    print("=" * 60)

    # Configuration
    rng = jax.random.PRNGKey(42)
    nu = 0.01  # Kinematic viscosity (Re = 100)
    n_epochs = 5000
    learning_rate = 1e-3

    print(f"\nConfiguration:")
    print(f"  Reynolds number: {1.0/nu:.1f}")
    print(f"  Kinematic viscosity: {nu}")
    print(f"  Training epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")

    # Generate training data
    print("\nGenerating training data...")
    data = generate_training_data(rng, n_pde=2000, n_bc=100)
    print(f"  PDE points: {len(data['x_pde'])}")
    print(f"  BC points: {len(data['x_bc'])}")
    print(f"  IC points: {len(data['x_ic'])}")

    # Create model
    print("\nInitializing neural network...")
    model = NSNetwork(hidden_dim=128, num_layers=4)
    state = create_ns_train_state(rng, model, learning_rate=learning_rate)
    print(f"  Architecture: 3 inputs -> {model.num_layers}x{model.hidden_dim} -> 3 outputs")

    # Train
    print(f"\nTraining for {n_epochs} epochs...")
    start_time = time.time()
    state, history = train_ns_pinn(
        state,
        data["x_pde"],
        data["y_pde"],
        data["t_pde"],
        data["x_bc"],
        data["y_bc"],
        data["t_bc"],
        data["u_bc"],
        data["v_bc"],
        data["x_ic"],
        data["y_ic"],
        data["u_ic"],
        data["v_ic"],
        nu=nu,
        num_epochs=n_epochs,
        print_every=500,
    )
    elapsed_time = time.time() - start_time

    # Results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"  Time: {elapsed_time:.2f} seconds")
    print(f"  Final total loss: {history['total'][-1]:.6e}")
    print(f"  Final momentum loss: {history['momentum'][-1]:.6e}")
    print(f"  Final continuity loss: {history['continuity'][-1]:.6e}")
    print(f"  Final BC loss: {history['bc'][-1]:.6e}")

    # Visualize
    print("\nGenerating visualizations...")
    visualize_solution(state, nu, "navier_stokes_results.png")
    plot_training_history(history, "training_history.png")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    print("  - navier_stokes_results.png: Flow field visualization")
    print("  - training_history.png: Training curves")


if __name__ == "__main__":
    main()
