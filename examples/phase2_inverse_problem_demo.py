"""Demo: Inverse problem - estimate thermal conductivity from measurements.

Phase 2.2: Given temperature sensor data, find the hidden parameter α.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import linen as nn

from phio.physics.heat import heat_equation_residual
from phio.solvers.inverse_problem import InverseProblemSolver


class PINN_MLP(nn.Module):
    """Physics-Informed Neural Network."""

    @nn.compact
    def __call__(self, x, t):
        inputs = jnp.concatenate([x, t], axis=-1)
        x = nn.Dense(64)(inputs)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        return nn.Dense(1)(x)


def gaussian_ic(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(-50 * (x - 0.5) ** 2)


def analytical_solution(x: jnp.ndarray, t: jnp.ndarray, alpha: float) -> jnp.ndarray:
    sigma_t = jnp.sqrt(1 + 200 * alpha * t)
    return jnp.exp(-50 * (x - 0.5) ** 2 / (sigma_t**2)) / sigma_t


def generate_measurements(rng, true_alpha=0.01, n_measurements=30, noise_level=0.01):
    """Simulate experimental measurements with noise."""
    x_meas = jax.random.uniform(rng, shape=(n_measurements,), minval=0.1, maxval=0.9)
    t_meas = jax.random.uniform(rng, shape=(n_measurements,), minval=0.1, maxval=0.5)

    u_true = jax.vmap(lambda x, t: analytical_solution(x, t, true_alpha))(
        x_meas, t_meas
    )

    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, shape=u_true.shape) * noise_level
    u_meas = u_true + noise

    return x_meas, t_meas, u_meas, u_true


def plot_results(x_meas, t_meas, u_meas, u_true, history, true_alpha):
    """Visualize inverse problem results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(history["total"], label="Total Loss", linewidth=2)
    ax.plot(history["data"], label="Data Loss", linewidth=2, alpha=0.7)
    ax.plot(history["pde"], label="PDE Loss", linewidth=2, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Parameter estimation convergence
    ax = axes[0, 1]
    ax.plot(history["alpha"], linewidth=2, label="Estimated α")
    ax.axhline(
        y=true_alpha,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"True α = {true_alpha}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("α (Thermal Conductivity)")
    ax.set_title("Parameter Estimation Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Measurement locations
    ax = axes[1, 0]
    scatter = ax.scatter(
        x_meas, t_meas, c=u_meas, s=100, cmap="viridis", edgecolor="black"
    )
    ax.set_xlabel("Space (x)")
    ax.set_ylabel("Time (t)")
    ax.set_title(f"Measurement Locations (n={len(u_meas)})")
    plt.colorbar(scatter, ax=ax, label="Temperature u(x,t)")
    ax.grid(True, alpha=0.3)

    # Plot 4: Measurement fit
    ax = axes[1, 1]
    ax.scatter(u_true, u_meas, alpha=0.6, s=50, edgecolor="black")
    lim = [min(u_true.min(), u_meas.min()), max(u_true.max(), u_meas.max())]
    ax.plot(lim, lim, "r--", linewidth=2, label="Perfect Fit")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Measured Values (with noise)")
    ax.set_title("Measurement Quality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("inverse_problem_results.png", dpi=150, bbox_inches="tight")
    print("\n✅ Results saved to 'inverse_problem_results.png'")
    plt.show()


def main():
    """Run inverse problem demo."""
    print("=" * 70)
    print("PHASE 2.2: INVERSE PROBLEM - PARAMETER ESTIMATION")
    print("=" * 70)

    # Ground truth
    TRUE_ALPHA = 0.01
    NOISE_LEVEL = 0.01
    N_MEASUREMENTS = 30

    # Generate synthetic measurements
    rng = jax.random.PRNGKey(42)
    x_meas, t_meas, u_meas, u_true = generate_measurements(
        rng,
        true_alpha=TRUE_ALPHA,
        n_measurements=N_MEASUREMENTS,
        noise_level=NOISE_LEVEL,
    )

    print(f"\n🔬 Experimental Setup:")
    print(f"  True thermal conductivity: α = {TRUE_ALPHA}")
    print(f"  Measurement locations: {N_MEASUREMENTS}")
    print(f"  Noise level: ±{NOISE_LEVEL * 100:.1f}%")
    print(f"  Measurement range: x ∈ [0.1, 0.9], t ∈ [0.1, 0.5]")

    # Initial guess (deliberately wrong)
    INITIAL_GUESS = {"alpha": 0.05}  # 5x wrong!
    print(f"\n🎯 Initial guess: α = {INITIAL_GUESS['alpha']} (5x off!)")

    # Solve inverse problem
    model = PINN_MLP()
    solver = InverseProblemSolver(model, heat_equation_residual)

    state, estimated_params, history = solver.solve_inverse_problem(
        rng,
        x_meas,
        t_meas,
        u_meas,
        gaussian_ic,
        INITIAL_GUESS,
        n_epochs=1000,
        n_collocation_points=100,
        data_weight=10.0,
        print_every=200,
    )

    # Results
    initial_error = abs(INITIAL_GUESS["alpha"] - TRUE_ALPHA)
    final_error = abs(estimated_params["alpha"] - TRUE_ALPHA)
    improvement = (1 - final_error / initial_error) * 100

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"True α:           {TRUE_ALPHA:.6f}")
    print(f"Initial guess:    {INITIAL_GUESS['alpha']:.6f}")
    print(f"Final estimate:   {estimated_params['alpha']:.6f}")
    print(f"Initial error:    {initial_error:.6f}")
    print(f"Final error:      {final_error:.6f}")
    print(f"Improvement:      {improvement:.2f}%")
    print("=" * 70)

    # Visualize
    plot_results(x_meas, t_meas, u_meas, u_true, history, TRUE_ALPHA)


if __name__ == "__main__":
    main()
