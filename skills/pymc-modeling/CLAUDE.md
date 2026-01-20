# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a Claude Code skill (`pymc-modeling`) that provides guidance for Bayesian statistical modeling with PyMC v5+. It is loaded when users work on tasks involving:
- Bayesian inference and posterior sampling
- Hierarchical/multilevel models
- GLMs, time series, Gaussian processes, BART, mixture models
- Zero-inflated, hurdle, censored, ordinal, and robust regression models
- MCMC diagnostics, model comparison (LOO-CV, WAIC)
- Causal inference with `pm.do`/`pm.observe`

## Structure

```
SKILL.md              # Main skill content (model specification, inference, diagnostics workflow)
references/
  arviz.md            # Comprehensive ArviZ guide (expert workflow, all plots, interpretation)
  priors.md           # Prior selection guide (weakly informative defaults, prior predictive checks)
  inference.md        # MCMC backends (nutpie, NumPyro/JAX, PyMC NUTS) and approximate inference
  diagnostics.md      # Quick diagnostics reference, troubleshooting, common problems
  gotchas.md          # Common pitfalls (parameterization, performance issues)
  gp.md               # Gaussian processes (kernels, HSGP for large data, priors)
  timeseries.md       # Time series models (AR, random walk, structural, seasonality)
  bart.md             # BART (pymc-bart) usage, variable importance, partial dependence
  mixtures.md         # Mixture models (Gaussian mixtures, label switching, marginalization)
  specialized_likelihoods.md  # Zero-inflated, hurdle, censored, ordinal, robust regression
  custom_models.md    # Custom distributions (DensityDist, Potential, Simulator, CustomDist)
```

## Key Technical Guidance

When this skill is loaded, apply these principles:

**Inference**: Default to nutpie for sampling (2-5x faster than PyMC NUTS). Fall back to `pm.sample()` if nutpie unavailable. Use NumPyro/JAX for GPU or vectorized chains.

**Parameterization**: Prefer non-centered parameterization for hierarchical models with weak data to avoid divergences.

**Scalability**: Use `pm.gp.HSGP` instead of full GP for datasets > 500-1000 points. Avoid saving large Deterministics; use posterior predictive instead.

**Diagnostics**: Always check `r_hat < 1.01`, `ess_bulk/tail > 400`, and zero divergences. Use `az.plot_pair(..., divergences=True)` to identify problematic regions.

**ArviZ Workflow**: Follow the 5-phase expert workflow: (1) Immediate checks → (2) Deep convergence → (3) Model criticism → (4) Parameter interpretation → (5) Model comparison. Never interpret parameters until phases 1-3 pass. See `references/arviz.md` for comprehensive guidance.

## Editing This Skill

When modifying skill content:
- Keep code examples minimal and runnable
- Use coords/dims for interpretable InferenceData
- Include both "what" and "why" (e.g., why non-centered helps)
- Reference other files with relative links: `[references/priors.md](references/priors.md)`
