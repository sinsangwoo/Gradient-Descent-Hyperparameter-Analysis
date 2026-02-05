"""Demo: Multi-fidelity optimization for heat equation.

Phase 2.2 demonstration showing:
1. Low-fidelity training (coarse grid, fast)
2. High-fidelity refinement (fine grid, accurate)
3. Cost-effectiveness analysis
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import linen as nn

from phio.solvers.multifidelity import MultiFidelitySolver


class PINN_MLP(nn.Module):
    """Physics-Informed Neural Network with tanh activation."""

    hidden_dim: int = 64
    num_layers: int = 3

    @nn.compact
    def __call__(self, x, t):
        inputs = jnp.concatenate([x, t], axis=-1)
        for _ in range(self.num_layers):
            inputs = nn.Dense(self.hidden_dim)(inputs)
            inputs = nn.tanh(inputs)
        return nn.Dense(1)(inputs)


def gaussian_initial_condition(x: jnp.ndarray) -> jnp.ndarray:
    """Gaussian bump at x=0.5."""
    return jnp.exp(-50 * (x - 0.5) ** 2)


def analytical_heat_solution(
    x: jnp.ndarray, t: jnp.ndarray, alpha: float = 0.01
) -> jnp.ndarray:
    """Analytical solution for heat equation with Gaussian IC."""
    sigma_t = jnp.sqrt(1 + 200 * alpha * t)
    return jnp.exp(-50 * (x - 0.5) ** 2 / (sigma_t**2)) / sigma_t


def plot_results(results: dict):
    """Visualize multi-fidelity results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curves
    ax = axes[0, 0]
    low_history = results["low_fidelity"]["history"]
    high_history = results["high_fidelity"]["history"]

    ax.plot(low_history["total"], label="Low-Fidelity", linewidth=2)
    ax.plot(
        range(len(low_history["total"]), len(low_history["total"]) + len(high_history["total"])),
        high_history["total"],
        label="High-Fidelity",
        linewidth=2,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Accuracy comparison
    ax = axes[0, 1]
    fidelities = ["Low-Fidelity", "High-Fidelity"]
    errors = [
        results["low_fidelity"]["accuracy"]["relative_error"],
        results["high_fidelity"]["accuracy"]["relative_error"],
    ]
    colors = ["orange", "green"]

    bars = ax.bar(fidelities, errors, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Relative Error")
    ax.set_title("Accuracy Comparison")
    ax.set_ylim(0, max(errors) * 1.2)

    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{err:.4f}",
            ha="center",
            va="bottom",
        )

    # Plot 3: Time vs Accuracy trade-off
    ax = axes[1, 0]
    times = [
        results["low_fidelity"]["time"],
        results["total_time"],
    ]
    accuracies = [
        1 - results["low_fidelity"]["accuracy"]["relative_error"],
        1 - results["high_fidelity"]["accuracy"]["relative_error"],
    ]

    ax.scatter(times[0], accuracies[0], s=200, c="orange", marker="o", label="Low-Fidelity", edgecolor="black")
    ax.scatter(times[1], accuracies[1], s=200, c="green", marker="s", label="Multi-Fidelity", edgecolor="black")
    ax.plot(times, accuracies, "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Training Time (s)")
    ax.set_ylabel("Accuracy (1 - Relative Error)")
    ax.set_title("Cost-Effectiveness: Time vs Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Solution comparison
    ax = axes[1, 1]
    x_test = jnp.linspace(0, 1, 100)
    t_test = 0.5  # Final time

    # Ground truth
    u_true = jax.vmap(lambda x: analytical_heat_solution(x, t_test))(x_test)

    # Predictions
    low_state = results["low_fidelity"]["state"]
    high_state = results["high_fidelity"]["state"]

    u_low = jax.vmap(low_state.apply_fn, in_axes=(None, 0, 0))(
        low_state.params, x_test[:, None], jnp.full_like(x_test[:, None], t_test)
    ).squeeze()

    u_high = jax.vmap(high_state.apply_fn, in_axes=(None, 0, 0))(
        high_state.params, x_test[:, None], jnp.full_like(x_test[:, None], t_test)
    ).squeeze()

    ax.plot(x_test, u_true, "k-", label="Analytical", linewidth=2)
    ax.plot(x_test, u_low, "o-", label="Low-Fidelity", alpha=0.7, markersize=3)
    ax.plot(x_test, u_high, "s-", label="High-Fidelity", alpha=0.7, markersize=3)

    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=0.5)")
    ax.set_title("Solution at Final Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("multifidelity_results.png", dpi=150, bbox_inches="tight")
    print("\n✅ Results saved to 'multifidelity_results.png'")
    plt.show()


def main():
    """Run multi-fidelity optimization demo."""
    print("=" * 70)
    print("PHASE 2.2: MULTI-FIDELITY OPTIMIZATION DEMO")
    print("=" * 70)

    # Initialize
    rng = jax.random.PRNGKey(42)
    model = PINN_MLP(hidden_dim=64, num_layers=3)
    solver = MultiFidelitySolver(model, alpha=0.01)

    # Run multi-fidelity pipeline
    results = solver.multifidelity_pipeline(
        rng,
        gaussian_initial_condition,
        lambda x, t: analytical_heat_solution(x, t, alpha=0.01),
    )

    # Visualize
    plot_results(results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Low-Fidelity:  {results['low_fidelity']['time']:.2f}s, "
          f"Error: {results['low_fidelity']['accuracy']['relative_error']:.6f}")
    print(f"High-Fidelity: {results['high_fidelity']['time']:.2f}s, "
          f"Error: {results['high_fidelity']['accuracy']['relative_error']:.6f}")
    print(f"Total Time: {results['total_time']:.2f}s")
    print(f"Error Reduction: {results['error_reduction_percent']:.2f}%")
    print(f"Cost Function (Accuracy/sec): {results['cost_function']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
