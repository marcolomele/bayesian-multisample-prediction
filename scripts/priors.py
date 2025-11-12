"""
Prior distribution classes for Bayesian HPYP hyperparameter inference.

Provides log-probability density functions for priors on:
- Discount parameters (d_0, d_j): Uniform(0, 1)
- Strength parameters (θ_0, θ_j): Gamma(shape=300, rate=0.2)
"""

import numpy as np
from scipy.special import gammaln


class UniformPrior:
    """Uniform prior distribution on a bounded interval."""
    
    def __init__(self, lower: float = 0.0, upper: float = 1.0):
        """
        Initialize uniform prior.
        
        Args:
            lower: Lower bound
            upper: Upper bound
        """
        self.lower = lower
        self.upper = upper
        self.log_prob_value = -np.log(upper - lower)
    
    def log_pdf(self, x: float) -> float:
        """
        Compute log-probability density.
        
        Args:
            x: Value to evaluate
            
        Returns:
            Log-probability density
        """
        if self.lower <= x <= self.upper:
            return self.log_prob_value
        return -np.inf


class GammaPrior:
    """Gamma prior distribution with shape-rate parameterization."""
    
    def __init__(self, shape: float, rate: float):
        """
        Initialize Gamma prior.
        
        Args:
            shape: Shape parameter (α > 0)
            rate: Rate parameter (β > 0)
        """
        self.shape = shape
        self.rate = rate
        self.log_normalizer = shape * np.log(rate) - gammaln(shape)
    
    def log_pdf(self, x: float) -> float:
        """
        Compute log-probability density.
        
        Args:
            x: Value to evaluate (must be > 0)
            
        Returns:
            Log-probability density
        """
        if x <= 0:
            return -np.inf
        return self.log_normalizer + (self.shape - 1) * np.log(x) - self.rate * x

