# PhIO Examples

This directory contains example scripts and Jupyter notebooks demonstrating PhIO's capabilities.

---

## Quickstart

**File**: `quickstart.py`

Basic example showing the complete workflow for solving the 1D heat equation.

```bash
python examples/quickstart.py
```

**What you'll learn**:
- How to define a PDE
- Setting boundary and initial conditions
- Creating and training a PINN solver
- Evaluating against exact solutions

---

## Coming in Phase 1.3

### Heat Equation (1D)
**File**: `heat_equation_1d.py` (coming soon)

- Full training implementation
- Convergence analysis
- Comparison with Finite Difference Method
- Interactive visualization

### Navier-Stokes (2D)
**File**: `navier_stokes_2d.py` (coming soon)

- Lid-driven cavity flow
- Reynolds number sweep (Re = 100, 400, 1000)
- Vorticity and streamfunction plots
- Benchmark against CFD solutions

### Wave Equation (1D)
**File**: `wave_equation_1d.py` (coming soon)

- String vibration simulation
- d'Alembert solution comparison
- Energy conservation analysis

---

## Coming in Phase 2

### Inverse Problems
**Notebook**: `inverse_diffusion.ipynb`

- Parameter identification from noisy data
- Bayesian uncertainty quantification
- Sensor placement optimization

### Multi-Fidelity Optimization
**Notebook**: `multifidelity_heat.ipynb`

- Coarse PINN + Fine FDM hybrid
- 10x speedup demonstration
- Accuracy-cost trade-off analysis

---

## Running Examples

### Prerequisites

```bash
# Install PhIO with examples dependencies
pip install -e ".[dev]"
```

### Run Python Scripts

```bash
python examples/quickstart.py
```

### Run Jupyter Notebooks

```bash
jupyter notebook examples/
```

---

## Contributing Examples

Have a cool PhIO application? We'd love to include it!

1. Create your example in `examples/`
2. Add README entry above
3. Submit PR with tag `enhancement: examples`

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
