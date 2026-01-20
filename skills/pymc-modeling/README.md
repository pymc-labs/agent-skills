# PyMC Modeling Skill for Claude Code

A Claude Code skill that provides comprehensive guidance for Bayesian statistical modeling with PyMC v5+. When loaded, Claude gains expert knowledge of PyMC workflows, best practices, and common patterns.

When you work on PyMC modeling tasks, this skill is automatically loaded to provide:

- Modern PyMC v5+ API patterns
- MCMC inference best practices (nutpie, NumPyro/JAX backends)
- ArviZ diagnostic workflows
- Prior specification guidance
- Common model templates (hierarchical, GLMs, GPs, time series, BART, mixtures)

## Installation

Copy or symlink this directory to your Claude Code skills folder:

```bash
# Create skills directory if it doesn't exist
mkdir -p ~/.claude/skills

# Clone or copy this repository
git clone https://github.com/fonnesbeck/pymc-modeling.git ~/.claude/skills/pymc-modeling
```

The skill will automatically activate when Claude detects tasks involving Bayesian inference, PyMC, ArviZ, or related topics.

## Structure

```
SKILL.md                      # Main skill: model specification, inference, diagnostics
CLAUDE.md                     # Instructions for Claude when editing this skill
references/
  arviz.md                    # Comprehensive ArviZ guide (all plots, interpretation)
  priors.md                   # Prior selection (weakly informative defaults)
  inference.md                # MCMC backends (nutpie, NumPyro/JAX, approximate)
  diagnostics.md              # Quick diagnostics reference, troubleshooting
  gotchas.md                  # Common pitfalls and performance issues
  gp.md                       # Gaussian processes (HSGP, kernels, priors)
  timeseries.md               # Time series (AR, structural, seasonality)
  bart.md                     # BART (pymc-bart) usage and interpretation
  mixtures.md                 # Mixture models and label switching
  specialized_likelihoods.md  # Zero-inflated, censored, ordinal, robust
  custom_models.md            # Custom distributions (DensityDist, Potential)
```

## Topics Covered

### Core Workflow
- Model specification with coords/dims for interpretable InferenceData
- Centered vs non-centered parameterization
- Inference with nutpie (default), PyMC NUTS, and NumPyro/JAX
- Systematic 5-phase ArviZ diagnostic workflow
- Prior and posterior predictive checks
- Model comparison (LOO-CV, WAIC)
- Saving/loading results

### Model Types
- **Hierarchical/multilevel models**: Partial pooling, non-centered parameterization
- **GLMs**: Logistic, Poisson, negative binomial regression
- **Gaussian processes**: HSGP for scalability, periodic GPs, kernel selection
- **Time series**: AR, random walk, structural time series, seasonality
- **BART**: Bayesian additive regression trees via pymc-bart
- **Mixture models**: Gaussian mixtures, label switching solutions
- **Specialized likelihoods**: Zero-inflated, hurdle, censored, ordinal, robust

### Advanced Topics
- Custom distributions with `pm.DensityDist` and `pm.CustomDist`
- Soft constraints with `pm.Potential`
- Simulation-based inference with `pm.Simulator`
- Causal inference with `pm.do` and `pm.observe`
- pymc-extras for marginalized mixtures and R2D2 priors

## Key Recommendations

When this skill is loaded, Claude will follow these principles:

| Topic | Recommendation |
|-------|----------------|
| **Sampler** | nutpie by default (2-5x faster), PyMC NUTS as fallback |
| **Parameterization** | Non-centered for hierarchical models with weak data |
| **GPs** | HSGP for n > 500 points; full GP only for small datasets |
| **Diagnostics** | Check r_hat < 1.01, ESS > 400, zero divergences before interpretation |
| **Notebooks** | marimo preferred over Jupyter |

## Requirements

The skill assumes these packages are available:

- `pymc >= 5.0`
- `arviz`
- `nutpie` (recommended)
- `pymc-bart` (for BART models)
- `pymc-extras` (for specialized distributions)

Optional for GPU/JAX acceleration:
- `numpyro`
- `jax`

## License

This skill is provided as-is for use with Claude Code.
