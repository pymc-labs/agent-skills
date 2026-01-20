# Common Pitfalls

## Statistical Issues

### Centered Parameterization with Weak Data

```python
# Causes divergences with few observations per group
alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, dims="group")  # BAD

# Non-centered parameterization
alpha_offset = pm.Normal("alpha_offset", 0, 1, dims="group")
alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset, dims="group")
```

### Flat Priors on Scale Parameters

```python
# Problematic in hierarchical models
sigma = pm.Uniform("sigma", 0, 100)  # BAD
sigma = pm.HalfFlat("sigma")  # BAD

# Weakly informative alternatives
sigma = pm.HalfNormal("sigma", sigma=1)
sigma = pm.HalfCauchy("sigma", beta=1)
sigma = pm.Exponential("sigma", lam=1)
```

### Label Switching in Mixture Models

```python
# Unordered components cause label switching
mu = pm.Normal("mu", 0, 10, dims="component")  # BAD

# Order constraint
mu_raw = pm.Normal("mu_raw", 0, 10, dims="component")
mu = pm.Deterministic("mu", pt.sort(mu_raw), dims="component")
```

### Missing Prior Predictive Checks

Always check prior implications before fitting:

```python
with model:
    prior_pred = pm.sample_prior_predictive()

az.plot_ppc(prior_pred, group="prior")
```

## Performance Issues

### Full GP on Large Datasets

```python
# O(n³) - slow for n > 1000
gp = pm.gp.Marginal(cov_func=cov)
y = gp.marginal_likelihood("y", X=X_large, y=y_obs)

# O(nm) - use HSGP instead
gp = pm.gp.HSGP(m=[30], c=1.5, cov_func=cov)
f = gp.prior("f", X=X_large)
```

### Saving Large Deterministics

```python
# Stores n_obs x n_draws array
mu = pm.Deterministic("mu", X @ beta, dims="obs")  # SLOW

# Don't save intermediate computations
mu = X @ beta  # Not saved, use posterior_predictive if needed
```

### Recompiling for Each Dataset

```python
# Recompiles every iteration
for dataset in datasets:
    with pm.Model() as model:
        # ...
        idata = pm.sample()

# Use pm.Data to avoid recompilation
with pm.Model() as model:
    x = pm.Data("x", x_initial)
    # ...

for dataset in datasets:
    pm.set_data({"x": dataset["x"]})
    idata = pm.sample()
```

## Identifiability Issues

### Symptoms

- Strong parameter correlations in pair plots
- Very wide posteriors despite lots of data
- Different chains converging to different solutions
- R-hat > 1.01 despite long chains

### Common Causes

**Overparameterized models**: More parameters than the data can support.

```python
# Too many group-level effects for small groups
alpha_group = pm.Normal("alpha_group", 0, 1, dims="group")  # 100 groups, 3 obs each
beta_group = pm.Normal("beta_group", 0, 1, dims="group")    # Can't estimate both
```

**Multicollinearity**: Correlated predictors make individual effects unidentifiable.

**Redundant random effects**: Nested effects without constraints.

### Fixes

**Sum-to-zero constraints** for categorical effects:

```python
import pytensor.tensor as pt

# Constrain group effects to sum to zero
alpha_raw = pm.Normal("alpha_raw", 0, 1, shape=n_groups - 1)
alpha = pm.Deterministic("alpha", pt.concatenate([alpha_raw, -alpha_raw.sum(keepdims=True)]))
```

**QR decomposition** for regression with correlated predictors:

```python
# Orthogonalize design matrix
Q, R = np.linalg.qr(X)

with pm.Model() as qr_model:
    beta_tilde = pm.Normal("beta_tilde", 0, 1, dims="features")
    beta = pm.Deterministic("beta", pt.linalg.solve(R, beta_tilde))
    mu = Q @ beta_tilde  # Use Q directly in likelihood
```

**Reduce model complexity**: Start simple, add complexity only if needed.

### Diagnosis

```python
# Check for strong correlations
az.plot_pair(idata, var_names=["alpha", "beta"], divergences=True)

# Look for banana-shaped or ridge-like posteriors
# These indicate non-identifiability
```

## Prior-Data Conflict

### Symptoms

- Posterior piled against prior boundary
- Prior and posterior distributions look very different
- Divergences concentrated near prior boundaries
- Effective sample size very low for some parameters

### Diagnosis

```python
# Compare prior and posterior
az.plot_dist_comparison(idata, var_names=["sigma"])

# Visual comparison for all parameters
fig, axes = plt.subplots(1, len(param_names), figsize=(4*len(param_names), 3))
for ax, var in zip(axes, param_names):
    az.plot_density(idata.prior, var_names=[var], ax=ax, colors="C0", label="Prior")
    az.plot_density(idata.posterior, var_names=[var], ax=ax, colors="C1", label="Posterior")
    ax.set_title(var)
```

### Common Scenarios

**Prior too narrow**: Data suggests values outside prior range.

```python
# Prior rules out likely values
sigma = pm.HalfNormal("sigma", sigma=0.1)  # If true sigma is ~5, this fights the data

# Fix: Use domain knowledge, not convenience
sigma = pm.HalfNormal("sigma", sigma=5)  # Allow for larger values
```

**Prior on wrong scale**: Common when using default priors without checking.

```python
# Default prior on standardized scale
beta = pm.Normal("beta", 0, 1)  # Fine if X is standardized

# But if X ranges from 10000 to 50000...
# Standardize predictors or adjust prior
X_scaled = (X - X.mean()) / X.std()
```

### Resolution

1. Check data for errors (outliers, coding mistakes)
2. Reconsider prior based on domain knowledge
3. Use prior predictive checks to validate
4. If justified, use more flexible prior

```python
# Validate prior choice
with model:
    prior_pred = pm.sample_prior_predictive()

# Check if prior predictive includes observed data range
prior_y = prior_pred.prior_predictive["y"].values.flatten()
print(f"Prior range: [{prior_y.min():.1f}, {prior_y.max():.1f}]")
print(f"Data range: [{y_obs.min():.1f}, {y_obs.max():.1f}]")
```

## Multicollinearity

### The Problem

Correlated predictors make individual coefficient estimates unstable, even though predictions remain valid.

### Detection

```python
import numpy as np

# Condition number (>30 suggests problems)
condition_number = np.linalg.cond(X)
print(f"Condition number: {condition_number:.1f}")

# Correlation matrix
import pandas as pd
corr = pd.DataFrame(X, columns=feature_names).corr()
print(corr)

# Variance inflation factors (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = feature_names
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print(vif_data)  # VIF > 5-10 indicates multicollinearity
```

### Symptoms in Posteriors

```python
# Strong negative correlation between coefficients
az.plot_pair(idata, var_names=["beta"])
# Look for elongated ellipses or banana shapes

# Wide credible intervals despite large N
summary = az.summary(idata, var_names=["beta"])
print(summary[["mean", "sd", "hdi_3%", "hdi_97%"]])
```

### Solutions

**Drop redundant predictors**:

```python
# If age and birth_year are both included, drop one
X = X[:, [i for i, name in enumerate(feature_names) if name != "birth_year"]]
```

**Use regularizing priors**:

```python
# Ridge-like prior (shrinks toward zero)
beta = pm.Normal("beta", mu=0, sigma=0.5, dims="features")

# Horseshoe prior (sparse, some coefficients near zero)
import pymc_extras as pmx
# pmx.Horseshoe(...)
```

**QR parameterization** (orthogonalizes predictors):

```python
Q, R = np.linalg.qr(X)
R_inv = np.linalg.inv(R)

with pm.Model() as model:
    # Sample in orthogonal space
    theta = pm.Normal("theta", 0, 1, dims="features")

    # Transform back to original scale
    beta = pm.Deterministic("beta", pt.dot(R_inv, theta))

    # Likelihood uses Q (orthogonal)
    mu = pt.dot(Q, theta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

**Interpret carefully**: If prediction is the goal, multicollinearity may not matter—just don't interpret individual coefficients.
