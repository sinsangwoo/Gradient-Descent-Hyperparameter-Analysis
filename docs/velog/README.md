# Velog Technical Blog Posts

This directory contains technical blog posts for publication on Velog.

## Contents

### Main Post
- **`phio-technical-report.md`**: Comprehensive technical report
  - Academic paper style
  - 8 sections (Abstract to Conclusion)
  - Code examples and equations
  - 5 figures with detailed captions

### Supporting Files
- **`generate_figures.py`**: Script to generate all figures
- **`korean-summary.md`**: Korean guide for posting
- **`figures/`**: Generated PNG files (created by script)

## Quick Start

### 1. Generate Figures

```bash
cd docs/velog
python generate_figures.py
```

**Output:**
```
figures/
├── fig1_training_loss.png
├── fig2_benchmark_comparison.png
├── fig3_multi_gpu_speedup.png
├── fig4_error_distribution.png
└── fig5_performance_comparison.png
```

### 2. Upload to Velog

1. Go to https://velog.io/write
2. Upload all 5 figures
3. Copy image URLs
4. Replace placeholders in `phio-technical-report.md`
5. Paste content and publish

### 3. Recommended Tags

```
#PhysicsInformedNeuralNetworks
#MachineLearning
#ComputationalFluidDynamics
#JAX
#MultiGPU
#OpenSource
#ProductionML
```

## Post Structure

### Section Overview

| Section | Content | Length |
|---------|---------|--------|
| Abstract | Summary and key results | 1 paragraph |
| Introduction | Motivation and contributions | 2-3 paragraphs |
| Related Work | Prior art and comparisons | 2-3 paragraphs |
| Methods | PINN formulation, multi-GPU | 3-4 code blocks |
| Experiments | Setup and configuration | 1-2 tables |
| Results | Accuracy and performance | 5 figures |
| Discussion | Findings and limitations | 2-3 paragraphs |
| Conclusion | Summary and future work | 1 paragraph |

### Key Metrics Highlighted

- **Speed**: 34x faster than OpenFOAM (4 GPUs)
- **Accuracy**: < 5% error vs Ghia benchmark
- **Scalability**: 85% parallel efficiency (4 GPUs)
- **Deployment**: Docker + REST API + Dashboard

## Customization

### Modify Figures

Edit `generate_figures.py` to customize:
- Colors and styles
- Data points
- Labels and titles
- Resolution (default: 150 DPI)

### Update Content

Edit `phio-technical-report.md`:
- Add your own results
- Include additional experiments
- Update references
- Adjust formatting

## Publication Checklist

- [ ] Generate all figures
- [ ] Upload images to Velog
- [ ] Replace placeholder URLs
- [ ] Proofread content
- [ ] Check code syntax highlighting
- [ ] Verify all links
- [ ] Add tags
- [ ] Set series (optional)
- [ ] Publish
- [ ] Share on social media
- [ ] Update GitHub README with link

## Follow-up Posts

Suggested series:

1. **Technical Report** (this post)
2. Multi-GPU Training Deep Dive
3. PINN Deployment Tutorial
4. CFD Benchmark Guide
5. Production ML Best Practices

## License

MIT License - Feel free to adapt for your own projects.

## Contact

For questions or collaborations:
- GitHub: https://github.com/sinsangwoo/Gradient-Descent-Hyperparameter-Analysis
- Issues: https://github.com/sinsangwoo/Gradient-Descent-Hyperparameter-Analysis/issues
