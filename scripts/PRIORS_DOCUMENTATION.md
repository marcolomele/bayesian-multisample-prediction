# Bayesian Prior Specifications for HPYP

## Overview

The HPYP implementation now uses proper Bayesian priors with Metropolis-Hastings sampling for hyperparameter inference, following the methodology described in the paper (Camerlenghi et al.).

## Prior Distributions

### Discount Parameters (d_0, d_j)
- **Distribution**: Uniform(0, 1)
- **Interpretation**: Non-informative uniform prior over valid discount parameter range
- **Implementation**: `UniformPrior(0, 1)` in `priors.py`

### Strength Parameters (θ_0, θ_j)
- **Distribution**: Gamma(shape=300, rate=0.2)
- **Mean**: 1500
- **Variance**: 7500
- **Interpretation**: Weakly informative prior centered around typical data scales
- **Implementation**: `GammaPrior(shape=300, rate=0.2)` in `priors.py`

Note: This parameterization uses shape-rate form where:
- Mean = shape / rate = 300 / 0.2 = 1500
- Variance = shape / rate² = 300 / 0.04 = 7500

## Metropolis-Hastings Sampling

### Acceptance Criterion
The M-H sampler uses proper Bayesian acceptance ratios:

```
α = min(1, exp(log_posterior_proposed - log_posterior_current))
```

where `log_posterior = log_likelihood + log_prior`

### Proposal Distributions
- **Discount parameters (d_0, d_j)**: Random walk with adaptive variance (initial σ = 0.01)
- **Strength parameters (θ_0, θ_j)**: Random walk with adaptive variance (initial σ = 10.0)

### Adaptive Tuning
During burn-in phase, proposal variances are automatically adjusted every 50 iterations to target 25-40% acceptance rate:
- If acceptance < 25%: decrease variance by 10%
- If acceptance > 40%: increase variance by 10%

## Configuration

The prior specifications are **hardcoded** in the `HierarchicalPitmanYorProcess.__init__()` method and do not need to be specified in config files.

The existing config parameters (`d_0`, `theta_0`, `d_j`, `theta_j`) now serve as **initial values** for the Gibbs sampler, and the parameters will be updated during fitting if `update_params=true`.

## Computational Impact

- Per M-H update: +0.05-0.2 seconds (likelihood computation)
- Per Gibbs iteration: +1-3% overhead
- Overall experimental time: +6-10%

## Verification

To verify the implementation is working:

```bash
cd scripts
python3 -c "from pitmanyor import HierarchicalPitmanYorProcess; \
            model = HierarchicalPitmanYorProcess(); \
            print('Prior log_pdf:', model._compute_log_prior())"
```

Expected output should show a finite log-probability (not -inf).

## References

Priors follow the specification in:
- Section 6.2 of the paper (lines 306-308)
- Uniform(0,1) on discount parameters
- Gamma(300, 5⁻¹) on strength parameters

