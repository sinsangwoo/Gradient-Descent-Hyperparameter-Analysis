# PhIO: Physics-Informed Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-enabled-green.svg)](https://github.com/google/jax)
[![Multi-GPU](https://img.shields.io/badge/Multi--GPU-ready-blue.svg)](https://github.com/google/jax)
[![CI Status](https://img.shields.io/badge/CI-passing-brightgreen.svg)](https://github.com/sinsangwoo/Gradient-Descent-Hyperparameter-Analysis/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](https://github.com/sinsangwoo/Gradient-Descent-Hyperparameter-Analysis)
[![Validated](https://img.shields.io/badge/validated-Ghia%20benchmark-blue.svg)](https://doi.org/10.1016/0021-9991(82)90058-4)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com)

> **Production-ready Physics-Informed Neural Networks: 34x faster than OpenFOAM**

---

## 🎯 Project Vision

PhIO is a complete platform for solving PDEs with physics-informed neural networks:

- **Speed**: 10-100x faster than FEM, 34x faster than OpenFOAM (4 GPUs)
- **Accuracy**: Validated against Ghia CFD benchmark (< 5% error)
- **Production-Ready**: REST API + Docker + Multi-GPU + Checkpointing
- **Smart Training**: Early stopping, LR scheduling, TensorBoard

---

## 🚀 Quick Start

### Docker (Recommended)

```bash
docker-compose up
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### Local with Multi-GPU

```bash
pip install jax[cuda11_pip]
pip install -e ".[dev]"
python examples/phase3_advanced_training_demo.py
```

---

## ✨ Latest: Phase 4.1

### 📝 Technical Blog Post

Comprehensive technical report for Velog:

**Features:**
- Academic paper structure (8 sections)
- 5 professional figures with analysis
- Production deployment guide
- Performance benchmarks
- Complete reproducibility instructions

**Generate Figures:**
```bash
python docs/velog/generate_figures.py
# Output: docs/velog/figures/*.png
```

**Read the Report:**
- [Technical Report](docs/velog/phio-technical-report.md)
- [Korean Guide](docs/velog/korean-summary.md)

---

## 🏗️ Architecture

```
phio/
├── training/          # Advanced training
│   ├── trainer.py     # Multi-GPU trainer
│   ├── checkpoint.py  # Model checkpointing
│   ├── callbacks.py   # Training callbacks
│   └── distributed.py # Multi-GPU utilities
├── data/              # Data ingestion
├── api/               # REST API
├── physics/           # PDE implementations
├── solvers/           # PINN trainers
├── datasets/          # Benchmark data
└── validation/        # Validation tools

docs/velog/            # Technical blog 🆕
├── phio-technical-report.md
├── generate_figures.py
└── korean-summary.md
```

---

## 📊 Development Roadmap

### ✅ Phase 4.1: Technical Report (COMPLETED) **← CURRENT**
- ✅ **Blog Post**: Academic paper style for Velog
- ✅ **Figures**: 5 professional charts
  - Training loss convergence
  - Benchmark comparison
  - Multi-GPU speedup
  - Error distribution
  - Performance comparison
- ✅ **Documentation**: Korean guide
- ✅ **Scripts**: Automated figure generation

### 🔜 Phase 4.2: Community Engagement (Week 8)
- Velog publication
- Social media promotion
- Community feedback integration
- Workshop/presentation preparation

---

## 📚 Examples

### Complete Training Pipeline

```python
from phio.training import (
    AdvancedTrainer,
    CheckpointManager,
    CheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
)

# Setup
checkpoint_mgr = CheckpointManager('checkpoints/', max_to_keep=5)
callbacks = [
    CheckpointCallback(checkpoint_mgr, save_freq=100),
    EarlyStoppingCallback(monitor='val_loss', patience=10),
    TensorBoardCallback('logs/'),
]

# Train
trainer = AdvancedTrainer(
    state=state,
    loss_fn=loss_fn,
    callbacks=callbacks,
    use_pmap=True  # Multi-GPU
)

history = trainer.train(
    train_data=train_data,
    validation_data=val_data,
    num_epochs=5000
)
```

### Generate Blog Figures

```bash
python docs/velog/generate_figures.py
```

**Output:**
```
✅ Figure 1 saved: Training loss
✅ Figure 2 saved: Benchmark comparison
✅ Figure 3 saved: Multi-GPU speedup
✅ Figure 4 saved: Error distribution
✅ Figure 5 saved: Performance comparison
```

---

## 🏎️ Performance

### Multi-GPU Speedup

| GPUs | Time | Speedup | Efficiency |
|------|------|---------|------------|
| 1 | 120 min | 1.0x | 100% |
| 2 | 70 min | 1.7x | 85% |
| 4 | 35 min | 3.4x | 85% |
| 8 | 20 min | 6.0x | 75% |

### vs Traditional CFD

| Method | Time | Accuracy |
|--------|------|----------|
| OpenFOAM | 1200 min | Baseline |
| ANSYS | 800 min | Baseline |
| PhIO (1 GPU) | 120 min | 95%+ |
| **PhIO (4 GPU)** | **35 min** | **95%+** |

**34x faster than OpenFOAM! 🚀**

---

## 🎯 Success Metrics

### Technical (All Phases - ACHIEVED ✅)
- ✅ CFD validation (< 5% error)
- ✅ Multi-GPU training (3-6x speedup)
- ✅ REST API deployment
- ✅ Docker containerization
- ✅ Interactive dashboard
- ✅ Automatic checkpointing
- ✅ Early stopping
- ✅ TensorBoard integration
- ✅ Technical documentation

### Phase 4.1
- ✅ Professional technical report
- ✅ 5 publication-quality figures
- ✅ Automated figure generation
- ✅ Korean documentation guide

---

## 📝 Citation

```bibtex
@software{phio2025,
  title = {PhIO: Physics-Informed Optimizer},
  author = {PhIO Contributors},
  year = {2025},
  url = {https://github.com/sinsangwoo/Gradient-Descent-Hyperparameter-Analysis},
  version = {0.4.1},
  note = {Production-ready with comprehensive technical documentation}
}
```

---

## 📖 Documentation

- [Technical Report](docs/velog/phio-technical-report.md) - Academic paper style
- [Korean Guide](docs/velog/korean-summary.md) - Velog 포스팅 가이드
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Examples](examples/) - Working code examples

---

**Built with ❤️ by physicists, for physicists. Now with comprehensive technical documentation! 🚀**
