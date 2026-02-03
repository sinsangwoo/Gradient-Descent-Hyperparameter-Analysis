"""Integration test for heat equation solver."""

import pytest
import jax.numpy as jnp
from phio.physics import HeatEquation1D
from phio.solvers import PINNSolver
from phio.core import DirichletBC, InitialCondition


@pytest.mark.slow
class TestHeatEquationSolver:
    """Integration test for solving 1D heat equation."""

    def test_solver_initialization(self):
        """Test solver can be initialized."""
        pde = HeatEquation1D(domain=(0, 1), diffusion_coeff=0.01)
        solver = PINNSolver(pde, hidden_dims=[32, 32])
        
        assert solver.pde == pde
        assert solver.hidden_dims == [32, 32]

    def test_boundary_and_initial_conditions(self):
        """Test setting boundary and initial conditions."""
        pde = HeatEquation1D()
        solver = PINNSolver(pde)
        
        # Set BCs
        bc_left = DirichletBC("left", lambda t: 0.0)
        bc_right = DirichletBC("right", lambda t: 0.0)
        solver.set_boundary_conditions([bc_left, bc_right])
        
        # Set IC
        ic = InitialCondition(lambda x: jnp.sin(jnp.pi * x))
        solver.set_initial_condition(ic)
        
        assert len(solver.boundary_conditions) == 2
        assert solver.initial_condition is not None

    @pytest.mark.skip(reason="Training not fully implemented yet")
    def test_training_reduces_loss(self):
        """Test that training reduces loss over time."""
        pde = HeatEquation1D()
        solver = PINNSolver(pde, hidden_dims=[16, 16])
        
        # Setup problem
        bc_left = DirichletBC("left", lambda t: 0.0)
        bc_right = DirichletBC("right", lambda t: 0.0)
        solver.set_boundary_conditions([bc_left, bc_right])
        ic = InitialCondition(lambda x: jnp.sin(jnp.pi * x))
        solver.set_initial_condition(ic)
        
        # Train for small number of epochs
        results = solver.train(num_epochs=100, n_collocation=100)
        
        # Check loss decreased
        initial_loss = results["loss_history"][0]
        final_loss = results["loss_history"][-1]
        assert final_loss < initial_loss
