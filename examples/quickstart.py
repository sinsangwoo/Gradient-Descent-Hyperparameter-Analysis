"""Quickstart example: Solving 1D heat equation with PhIO.

This demonstrates the basic workflow:
1. Define PDE and domain
2. Set boundary/initial conditions
3. Create solver
4. Train PINN
5. Evaluate and visualize
"""

import jax.numpy as jnp
from phio.physics import HeatEquation1D
from phio.solvers import PINNSolver
from phio.core import DirichletBC, InitialCondition
from phio.utils import logger, compute_metrics


def main():
    logger.info("=" * 60)
    logger.info("PhIO Quickstart: 1D Heat Equation")
    logger.info("=" * 60)

    # 1. Define PDE
    logger.info("\n[Step 1] Defining heat equation: u_t = alpha * u_xx")
    pde = HeatEquation1D(
        domain=(0, 1),  # x ∈ [0, 1]
        time_domain=(0, 1),  # t ∈ [0, 1]
        diffusion_coeff=0.01,  # alpha = 0.01
    )
    logger.info(f"  Domain: x ∈ {pde.domain}, t ∈ {pde.time_domain}")
    logger.info(f"  Diffusion coefficient: α = {pde.alpha}")

    # 2. Set boundary conditions: u(0,t) = u(1,t) = 0
    logger.info("\n[Step 2] Setting boundary conditions")
    bc_left = DirichletBC(location="left", value_fn=lambda t: 0.0)
    bc_right = DirichletBC(location="right", value_fn=lambda t: 0.0)
    logger.info("  Left BC: u(0, t) = 0")
    logger.info("  Right BC: u(1, t) = 0")

    # 3. Set initial condition: u(x,0) = sin(π*x)
    logger.info("\n[Step 3] Setting initial condition")
    ic = InitialCondition(value_fn=lambda x: jnp.sin(jnp.pi * x))
    logger.info("  IC: u(x, 0) = sin(π·x)")

    # 4. Create PINN solver
    logger.info("\n[Step 4] Creating PINN solver")
    solver = PINNSolver(
        pde=pde,
        hidden_dims=[64, 64, 64],  # 3 hidden layers with 64 neurons each
        activation="tanh",  # Smooth activation for physics
        optimizer="adam",
        learning_rate=1e-3,
        seed=42,
    )
    solver.set_boundary_conditions([bc_left, bc_right])
    solver.set_initial_condition(ic)
    logger.info("  Network: [2] → [64] → [64] → [64] → [1]")
    logger.info("  Activation: tanh")
    logger.info("  Optimizer: Adam (lr=1e-3)")

    # 5. Train (NOTE: Full implementation in Phase 1.3)
    logger.info("\n[Step 5] Training PINN")
    logger.info("  Note: Full training implementation coming in Phase 1.3")
    logger.info("  This quickstart demonstrates the API design")

    # Placeholder for actual training
    # results = solver.train(
    #     num_epochs=10000,
    #     n_collocation=1000,
    #     n_boundary=100,
    #     n_initial=100,
    #     log_frequency=1000,
    # )

    # 6. Evaluate (placeholder)
    logger.info("\n[Step 6] Evaluation")
    logger.info("  Evaluation will compute L2 error vs exact solution")
    logger.info("  Expected target: L2 error < 1e-3")

    # Example of what evaluation will look like:
    # x_test = jnp.linspace(0, 1, 100)
    # t_test = jnp.linspace(0, 1, 100)
    # X_test, T_test = jnp.meshgrid(x_test, t_test)
    # u_pred = solver.predict(X_test.flatten(), T_test.flatten()).reshape(X_test.shape)
    # u_exact = pde.exact_solution(X_test, T_test)
    # metrics = compute_metrics(u_pred, u_exact)
    # logger.info(f"  L2 relative error: {metrics['l2_relative']:.2e}")

    logger.info("\n" + "=" * 60)
    logger.info("Quickstart completed! See examples/ for more tutorials.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
