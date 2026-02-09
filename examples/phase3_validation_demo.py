"""Phase 3.1: Real-world validation against Ghia benchmark.

Demonstrates:
1. Training PINN on lid-driven cavity
2. Comparing with Ghia et al. (1982) benchmark data
3. Computing quantitative error metrics
4. Generating comprehensive validation reports
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn

from phio.datasets.ghia_cavity import GhiaCavityData
from phio.physics.navier_stokes import lid_driven_cavity_bc
from phio.solvers.ns_pinn import create_ns_train_state, train_ns_pinn
from phio.validation.metrics import compute_error_metrics, generate_error_report
from phio.validation.visualize import (
    plot_benchmark_comparison,
    plot_error_distribution,
    plot_validation_dashboard,
)


class NSNetwork(nn.Module):
    """Neural network for Navier-Stokes."""

    hidden_dim: int = 128
    num_layers: int = 4

    @nn.compact
    def __call__(self, x, y, t):
        inputs = jnp.concatenate([x, y, t], axis=-1)

        h = inputs
        for _ in range(self.num_layers):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.tanh(h)

        out = nn.Dense(3)(h)
        return out


def generate_training_data(
    rng: jax.random.PRNGKey,
    n_pde: int = 2000,
    n_bc: int = 100,
) -> dict:
    """Generate collocation points."""
    rng, key = jax.random.split(rng)
    x_pde = jax.random.uniform(key, (n_pde,))
    y_pde = jax.random.uniform(key, (n_pde,))
    t_pde = jnp.zeros(n_pde)

    # Boundary points
    x_top = jnp.linspace(0, 1, n_bc)
    y_top = jnp.ones(n_bc)
    x_bottom = jnp.linspace(0, 1, n_bc)
    y_bottom = jnp.zeros(n_bc)
    x_left = jnp.zeros(n_bc)
    y_left = jnp.linspace(0, 1, n_bc)
    x_right = jnp.ones(n_bc)
    y_right = jnp.linspace(0, 1, n_bc)

    x_bc = jnp.concatenate([x_top, x_bottom, x_left, x_right])
    y_bc = jnp.concatenate([y_top, y_bottom, y_left, y_right])
    t_bc = jnp.zeros_like(x_bc)

    u_bc, v_bc = lid_driven_cavity_bc(x_bc, y_bc, u_lid=1.0)

    # Initial condition
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


def validate_pinn(
    state,
    reynolds_number: int,
    output_dir: str = ".",
):
    """Validate PINN against Ghia benchmark.

    Args:
        state: Trained PINN model state
        reynolds_number: Reynolds number
        output_dir: Directory for output files
    """
    print("\n" + "=" * 60)
    print(f"VALIDATING AGAINST GHIA BENCHMARK (Re={reynolds_number})")
    print("=" * 60)

    # Compare with benchmark
    u_error, v_error, predictions = GhiaCavityData.compare_with_pinn(state, reynolds_number)

    print(f"\nQuick Summary:")
    print(f"  U-velocity MAE: {u_error:.6f}")
    print(f"  V-velocity MAE: {v_error:.6f}")

    # Compute detailed metrics
    u_metrics = compute_error_metrics(predictions["u_pred"], predictions["u_benchmark"])
    v_metrics = compute_error_metrics(predictions["v_pred"], predictions["v_benchmark"])

    # Generate report
    report = generate_error_report(u_metrics, v_metrics, reynolds_number)
    print(f"\n{report}")

    # Save report to file
    report_path = Path(output_dir) / f"validation_report_re{reynolds_number}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Benchmark comparison
    plot_benchmark_comparison(
        predictions,
        reynolds_number,
        str(Path(output_dir) / f"benchmark_comparison_re{reynolds_number}.png"),
    )

    # 2. Error distribution
    plot_error_distribution(
        predictions,
        str(Path(output_dir) / f"error_distribution_re{reynolds_number}.png"),
    )

    # 3. Comprehensive dashboard
    plot_validation_dashboard(
        predictions,
        u_metrics,
        v_metrics,
        reynolds_number,
        str(Path(output_dir) / f"validation_dashboard_re{reynolds_number}.png"),
    )

    return u_metrics, v_metrics


def main():
    """Main validation demonstration."""
    print("=" * 60)
    print("PHASE 3.1: REAL-WORLD VALIDATION")
    print("=" * 60)
    print("\nObjective: Validate PINN against Ghia et al. (1982) benchmark")
    print("Standard CFD validation dataset from literature")

    # Configuration
    rng = jax.random.PRNGKey(42)
    reynolds_number = 100  # Start with Re=100
    nu = 1.0 / reynolds_number
    n_epochs = 5000
    learning_rate = 1e-3

    print(f"\nConfiguration:")
    print(f"  Reynolds number: {reynolds_number}")
    print(f"  Kinematic viscosity: {nu:.6f}")
    print(f"  Training epochs: {n_epochs}")
    print(f"  Architecture: 128x4 MLP")

    # Generate training data
    print("\nGenerating training data...")
    data = generate_training_data(rng, n_pde=2000, n_bc=100)
    print(f"  PDE points: {len(data['x_pde'])}")
    print(f"  BC points: {len(data['x_bc'])}")

    # Create and train model
    print("\nTraining PINN...")
    model = NSNetwork(hidden_dim=128, num_layers=4)
    state = create_ns_train_state(rng, model, learning_rate=learning_rate)

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
    training_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"  Time: {training_time:.2f} seconds")
    print(f"  Final loss: {history['total'][-1]:.6e}")

    # Validate against benchmark
    u_metrics, v_metrics = validate_pinn(state, reynolds_number)

    # Final summary
    avg_rel_l2 = (u_metrics["relative_l2"] + v_metrics["relative_l2"]) / 2

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"\nKey Results:")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  U-velocity MAE: {u_metrics['mae']:.6f}")
    print(f"  V-velocity MAE: {v_metrics['mae']:.6f}")
    print(f"  Average Relative L2 Error: {avg_rel_l2:.4f} ({avg_rel_l2*100:.2f}%)")

    if avg_rel_l2 < 0.05:
        print(f"  ✅ EXCELLENT: Error < 5%")
    elif avg_rel_l2 < 0.10:
        print(f"  ✓ GOOD: Error < 10%")
    else:
        print(f"  ⚠ Needs improvement: Error > 10%")

    print("\nOutput files generated:")
    print(f"  - validation_report_re{reynolds_number}.txt")
    print(f"  - benchmark_comparison_re{reynolds_number}.png")
    print(f"  - error_distribution_re{reynolds_number}.png")
    print(f"  - validation_dashboard_re{reynolds_number}.png")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
