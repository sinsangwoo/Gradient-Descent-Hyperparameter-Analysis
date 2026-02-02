# Contributing to PhIO

Thank you for your interest in contributing to PhIO! We welcome contributions from physicists, machine learning researchers, and software engineers who want to advance physics-informed AI.

---

## 🎯 How to Contribute

### Types of Contributions

1. **Code**: New PDE solvers, optimizers, benchmarks
2. **Documentation**: Tutorials, API docs, examples
3. **Research**: Novel architectures, training techniques
4. **Testing**: Unit tests, integration tests, bug reports
5. **Community**: Blog posts, talks, educational content

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.9+
- Git
- CUDA-capable GPU (recommended, not required for development)

### Installation

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/physics-informed-optimizer.git
cd physics-informed-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies (Phase 1.2+)

```bash
# Code quality
pip install black flake8 mypy pylint

# Testing
pip install pytest pytest-cov pytest-xdist

# Documentation
pip install sphinx sphinx-rtd-theme nbsphinx

# Experiment tracking
pip install wandb tensorboard
```

---

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these additions:

- **Line length**: 100 characters (not 79)
- **Formatter**: [Black](https://github.com/psf/black) with default settings
- **Type hints**: Required for all public functions
- **Docstrings**: Google style

### Example

```python
from typing import Tuple
import jax.numpy as jnp


def solve_heat_equation(
    initial_condition: jnp.ndarray,
    time_steps: int,
    diffusion_coeff: float = 0.01,
) -> Tuple[jnp.ndarray, dict]:
    """Solve 1D heat equation using PINN.

    Args:
        initial_condition: Initial temperature distribution, shape (N,)
        time_steps: Number of temporal steps to simulate
        diffusion_coeff: Thermal diffusivity coefficient

    Returns:
        solution: Temperature at all time steps, shape (time_steps, N)
        metrics: Dictionary with loss history, runtime, etc.

    Raises:
        ValueError: If initial_condition is not 1D array

    Example:
        >>> u0 = jnp.sin(jnp.linspace(0, 2*jnp.pi, 100))
        >>> solution, metrics = solve_heat_equation(u0, time_steps=1000)
        >>> print(f"Final loss: {metrics['loss'][-1]:.6f}")
    """
    if initial_condition.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {initial_condition.shape}")

    # Implementation here...
    pass
```

### Pre-Commit Hooks

We use pre-commit to automatically enforce standards:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --extend-ignore=E203]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## 🧪 Testing Requirements

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=phio --cov-report=html

# Run specific test file
pytest tests/test_heat_equation.py

# Run tests in parallel
pytest -n auto
```

### Test Structure

```
tests/
├── unit/                 # Fast, isolated tests
│   ├── test_losses.py
│   ├── test_optimizers.py
│   └── test_utils.py
├── integration/          # Multi-component tests
│   ├── test_heat_solver.py
│   └── test_navier_stokes.py
└── benchmarks/           # Performance tests (CI skips)
    ├── test_speed.py
    └── test_accuracy.py
```

### Writing Good Tests

```python
import pytest
import jax.numpy as jnp
from phio.physics.heat import heat_equation_residual


class TestHeatEquation:
    """Test suite for 1D heat equation solver."""

    def test_residual_steady_state(self):
        """Residual should be zero for steady-state solution."""
        # Steady state: u(x) = constant
        u = jnp.ones(100)
        x = jnp.linspace(0, 1, 100)
        t = jnp.array(1.0)

        residual = heat_equation_residual(u, x, t, diffusion_coeff=0.01)
        assert jnp.allclose(residual, 0.0, atol=1e-6)

    def test_residual_shape(self):
        """Residual should have same shape as input."""
        u = jnp.ones((50, 100))  # (time, space)
        x = jnp.linspace(0, 1, 100)
        t = jnp.linspace(0, 1, 50)

        residual = heat_equation_residual(u, x, t)
        assert residual.shape == u.shape

    @pytest.mark.parametrize("diffusion_coeff", [0.001, 0.01, 0.1, 1.0])
    def test_residual_diffusion_coefficient(self, diffusion_coeff):
        """Residual should scale with diffusion coefficient."""
        # Test for different material properties
        pass  # Implementation...
```

### Coverage Requirements

- **Overall**: >80% line coverage
- **Core modules** (`phio/physics/`, `phio/solvers/`): >90% coverage
- **New PRs**: Must not decrease overall coverage

---

## 🔀 Pull Request Workflow

### 1. Create Feature Branch

```bash
# Always branch from main
git checkout main
git pull upstream main

# Use descriptive branch names
git checkout -b feature/adaptive-learning-rate
git checkout -b fix/boundary-condition-bug
git checkout -b docs/heat-equation-tutorial
```

### 2. Make Commits

**Commit Message Format** (following [Conventional Commits](https://www.conventionalcommits.org/)):

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring without functionality change
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance (dependencies, configs)

**Examples**:

```bash
git commit -m "feat(solver): Add curriculum learning scheduler for PINN training"

git commit -m "fix(heat): Correct boundary condition enforcement in Dirichlet case

Previously, ghost points were not properly updated, leading to
2nd-order accuracy loss. Now using symmetric extrapolation.

Fixes #42"

git commit -m "docs(examples): Add Jupyter notebook for inverse heat conduction"
```

### 3. Push and Create PR

```bash
# Push to your fork
git push origin feature/adaptive-learning-rate

# On GitHub, create Pull Request with template:
```

**PR Template**:

```markdown
## Description

Brief explanation of changes (2-3 sentences).

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to break)
- [ ] Documentation update

## Motivation and Context

Why is this change needed? What problem does it solve?
Link to related issues: Closes #XXX

## How Has This Been Tested?

- [ ] Unit tests pass (`pytest tests/unit`)
- [ ] Integration tests pass (`pytest tests/integration`)
- [ ] Manually tested on [describe scenario]

## Screenshots (if applicable)

[Add images/plots showing before/after]

## Checklist

- [ ] My code follows the style guidelines (black, flake8, mypy)
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing unit tests pass locally
- [ ] I have updated documentation (docstrings, README, examples)
```

### 4. Code Review Process

**Reviewers will check**:
1. Code quality (readability, efficiency, correctness)
2. Test coverage (new code must be tested)
3. Documentation (public functions must have docstrings)
4. Performance (no unnecessary slowdowns)
5. Backward compatibility (breaking changes need discussion)

**Response time**:
- Initial review: Within 48 hours
- Follow-up reviews: Within 24 hours

**Approval criteria**:
- 2+ approvals from maintainers
- All CI checks passing
- No unresolved comments

---

## 🔬 Research Contributions

### Publishing Results

If your contribution leads to publishable research:

1. **Reproducibility**: Provide scripts to regenerate all figures/tables
2. **Benchmarking**: Compare against at least one baseline method
3. **Ablation Study**: Show which components contribute to performance
4. **Open Data**: Share datasets (or provide download scripts)

### Co-Authorship Policy

- Significant code contributors: Co-authorship on papers using PhIO
- Substantial intellectual contributions: Case-by-case discussion
- Acknowledgment: All contributors listed in paper acknowledgments

---

## 📚 Documentation Guidelines

### Docstring Example

```python
def train_pinn(
    pde_residual: Callable,
    boundary_conditions: Dict[str, Callable],
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    num_epochs: int = 10000,
) -> Tuple[jnp.ndarray, dict]:
    """Train physics-informed neural network to solve PDE.

    This function implements the standard PINN training loop with automatic
    differentiation to enforce PDE constraints. Supports custom optimizers
    and curriculum learning schedules.

    Args:
        pde_residual: Function computing PDE residual. Must have signature
            `residual(params, x, t) -> jnp.ndarray`
        boundary_conditions: Dictionary mapping boundary names to functions.
            Example: {"left": lambda t: jnp.sin(t), "right": lambda t: 0.0}
        optimizer: Name of optimizer ("adam", "sgd", "lbfgs")
        learning_rate: Initial learning rate (may decay during training)
        num_epochs: Maximum number of training iterations

    Returns:
        params: Trained neural network parameters (PyTree)
        metrics: Dictionary containing:
            - "loss_history": List of loss values at each epoch
            - "pde_loss": PDE residual component of final loss
            - "bc_loss": Boundary condition component of final loss
            - "runtime": Total training time in seconds

    Raises:
        ValueError: If optimizer name not recognized
        RuntimeError: If training diverges (loss > 1e6)

    Example:
        >>> def wave_eq_residual(params, x, t):
        ...     u = neural_net(params, x, t)
        ...     u_tt = jax.grad(jax.grad(u, argnums=2), argnums=2)
        ...     u_xx = jax.grad(jax.grad(u, argnums=1), argnums=1)
        ...     return u_tt - u_xx  # Wave equation: u_tt = u_xx
        >>>
        >>> bc = {"left": lambda t: 0.0, "right": lambda t: 0.0}
        >>> params, metrics = train_pinn(wave_eq_residual, bc, num_epochs=5000)
        >>> print(f"Final loss: {metrics['loss_history'][-1]:.2e}")
        Final loss: 3.42e-05

    References:
        [1] Raissi et al. (2019) "Physics-informed neural networks" JCP
        [2] Wang et al. (2021) "Understanding gradient pathologies" SISC
    """
    # Implementation...
    pass
```

---

## 🐛 Reporting Bugs

### Before Submitting

1. Search existing issues (open + closed)
2. Try latest `main` branch
3. Minimize example (remove unrelated code)

### Bug Report Template

```markdown
**Describe the bug**
Clear and concise description.

**To Reproduce**
Minimal code to reproduce:

```python
import phio
# Minimal example that triggers bug
```

**Expected behavior**
What should happen?

**Actual behavior**
What actually happens? Include full error traceback.

**Environment**
- OS: [e.g., Ubuntu 22.04, Windows 11, macOS 14]
- Python version: [e.g., 3.9.7]
- PhIO version: [e.g., 0.1.0 or commit hash]
- JAX version: [e.g., 0.4.20]
- GPU: [e.g., NVIDIA RTX 4090, or "CPU only"]

**Additional context**
Any other relevant information.
```

---

## 💬 Community Guidelines

### Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/) v2.1:

- Be respectful and inclusive
- Welcome diverse perspectives
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, show-and-tell
- **Discord** (future): Real-time chat
- **Slack** (future): Research collaborations

---

## 🏆 Recognition

### Contributors List

All contributors are acknowledged in:
- `CONTRIBUTORS.md` file
- Repository README
- Research papers (if applicable)

### Swag & Rewards (Future)

- **10+ merged PRs**: PhIO stickers + t-shirt
- **Major feature**: PhIO hoodie + featured blog post
- **Top contributor**: Conference travel grant

---

## 📬 Contact

- **Maintainer**: [Your Name] ([@username](https://github.com/username))
- **Email**: phio-dev@example.com
- **Office Hours**: Fridays 2-4 PM UTC (Google Meet link in Discord)

---

**Thank you for contributing to PhIO! Together, we're building the future of physics simulation. 🚀**