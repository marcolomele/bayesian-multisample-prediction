### Overview
The script will implement a complete experimental pipeline comparing **independent exchangeable** vs. **partially exchangeable (dependent)** predictions for species sampling problems using the Hierarchical Pitman-Yor Process (HPYP).

---

### **1. Data Preparation Module**

**Function: `load_and_split_data(data_path, train_ratio=0.8, random_seed=None)`**

- Load input data from various formats (CSV, Stata .dta, text files)
- Support multiple data structures:
  - Single sample (split into train/test)
  - Multiple samples (split each independently)
- Split data into:
  - **Fit set (80%)**: Used for learning the latent structure via Gibbs sampler
  - **Predict set (20%)**: Used for evaluating predictions
- Return structured data with:
  - `n1, n2`: Sample sizes for fit data
  - `m1, m2`: Sample sizes for predict data
  - `K_n1, K_n2`: Number of distinct species in fit data
  - Species frequency distributions

---

### **2. Model Fitting Module**

**Function: `fit_hpyp_model(fit_data, num_iterations=10000, burn_in=5000, model_type='dependent')`**

**2.1 Parameter Setup**
- Initialize hyperparameters:
  - For **independent model**: `(θ_1,0, δ_1,0, θ_1, δ_1)` and `(θ_2,0, δ_2,0, θ_2, δ_2)` separately
  - For **dependent model**: Single set `(θ_0, δ_0, θ, δ)` shared across samples
- Set priors:
  ```
  δ ~ U(0, 1)
  θ ~ Gamma(300, 1/5)
  ```

**2.2 Gibbs Sampler Implementation**
- Implement latent variable sampling for tables `T_i,j`:
  - Sample table assignments for observed data
  - Update base-level tables
  - Update group-level tables
  
- For each iteration `t = 1, ..., num_iterations`:
  - **Step 2.a**: Update table assignments `T_i,r` for existing customers
    - Use full conditionals from equation (12) in paper
    - Handle "new table" vs. "existing table" probabilities
  
  - Update hyperparameters `(θ_0, δ_0, θ, δ)` via Metropolis-Hastings
  
- Store posterior samples after burn-in
- Track:
  - Number of tables at base level (`l_••`)
  - Number of tables per group (`l_i•`)
  - Number of distinct species (`K_n`)

**Return:**
- Fitted model object with posterior samples
- Latent table structure
- Estimated parameters

---

### **3. Prediction Module**

**Function: `predict_new_species(fitted_model, fit_data, m_values, num_iterations=10000, model_type='dependent')`**

**3.1 Prediction Algorithm (Gibbs Sampler Part 2)**

For each additional sample size `m ∈ {200, 400, ..., 2000}`:
- For iteration `t = 1, ..., num_iterations`:
  - **Step 2.b**: Generate `(X_i,n_i+r, T_i,n_i+r)` for `r = 1, ..., m_i`
    - Use predictive distributions from paper:
      - `P(X_i,r = "new", T_i,r = "new" | ...)`: Completely new species
      - `P(X_i,r = X*_h, T_i,r = "new" | ...)`: Existing species, new table
      - `P(X_i,r = X*_h, T_i,r = T*_h,l | ...)`: Existing species, existing table

**3.2 Track Quantities of Interest**

For each sample `i = 1, 2`:
- **L^(0,0)_(i,m)**: Species new to both samples
- **L^(0,1)_(1,m)**: Species in sample 2 but new to sample 1
- **L^(1,0)_(2,m)**: Species in sample 1 but new to sample 2
- **L^0_(i,m) = L^(0,0)_(i,m) + L^(0,1)_(i,m)** (or with indices swapped): Total new species

For independent model:
- Only track **L^0_(i,m)**: Total new species per sample (no sharing)

**3.3 Compute Statistics**
- Posterior mean: `L̂^0_(i,m) = (1/T) Σ_t L^0_(i,m,t)`
- 95% HPD intervals using quantiles
- One-step prediction probabilities (m=1)
- Verify linearity property from Theorem 1

**Return:**
- Dictionary with predictions for each `m` value
- Posterior samples for uncertainty quantification

---

### **4. Comparison Module**

**Function: `compare_models(independent_results, dependent_results, m_values)`**

Compare independent vs. dependent predictions:
- **Borrowing of strength effect**: 
  - Show reduced uncertainty (narrower HPD intervals) in dependent model
  - Show convergence of predictions between samples in dependent model
  
- **Detection rates**:
  - Compare slopes of L̂^0_(i,m) vs. m (linearity check)
  - Show faster/slower detection rates
  
- **Shared species tracking** (dependent only):
  - L^(0,1)_(1,m) and L^(1,0)_(2,m): Species becoming shared

---

### **5. Visualization Module**

**Function: `create_visualizations(results, output_dir='./results')`**

**5.1 Main Figures (replicate Figure 1 and Figure 2 from paper)**
- **Plot A**: Independent model - Total new species vs. m
  - Two lines: Sample 1 and Sample 2
  - Shaded 95% HPD intervals
  - Show divergence between samples
  
- **Plot B**: Dependent model - Total new species vs. m
  - Two lines showing convergence
  - Narrower HPD intervals
  - Demonstrate borrowing of strength

**5.2 Additional Visualizations**
- Comparison of HPD interval widths
- Decomposition plots showing L^(0,0), L^(0,1), L^(1,0) components
- Parameter posterior distributions
- Number of tables over iterations (diagnostics)

---

### **6. Output Tables Module**

**Function: `create_output_tables(results, output_dir='./results')`**

**6.1 Replicate Tables from Paper**

**Table 2 equivalent** (Independent Exchangeable):
```
| m    | L̂^0_(1,m) | HPD (95%)    | L̂^0_(2,m) | HPD (95%)    |
|------|-----------|--------------|-----------|--------------|
| 200  | ...       | (...)        | ...       | (...)        |
| 400  | ...       | (...)        | ...       | (...)        |
| ...  | ...       | ...          | ...       | ...          |
```

**Table 4 equivalent** (Partially Exchangeable):
```
| m    | L̂^(0,0)_(1,m) | L̂^(0,1)_(1,m) | L̂^0_(1,m) | HPD | L̂^(0,0)_(2,m) | L̂^(1,0)_(2,m) | L̂^0_(2,m) | HPD |
```

**6.2 Parameter Estimates Table**
- Posterior means and credible intervals for all hyperparameters
- Compare independent vs. dependent estimates

**6.3 Model Comparison Table**
- Average HPD width reduction
- Predictive accuracy metrics (if ground truth available)

---

### **7. Main Execution Pipeline**

**Function: `run_experiment(data_path, config)`**

```python
def run_experiment(data_path, config):
    """
    Main experimental pipeline.
    
    Args:
        data_path: Path to input data
        config: Dictionary with experimental settings:
            - train_ratio: Default 0.8
            - num_fit_iterations: Default 10000
            - burn_in: Default 5000
            - num_predict_iterations: Default 10000
            - m_values: Default [200, 400, ..., 2000]
            - output_dir: Default './results'
    """
    
    # 1. Load and split data
    fit_data, predict_data = load_and_split_data(...)
    
    # 2. Fit independent models
    model_independent_1 = fit_hpyp_model(fit_data[0], model_type='independent')
    model_independent_2 = fit_hpyp_model(fit_data[1], model_type='independent')
    
    # 3. Fit dependent model
    model_dependent = fit_hpyp_model(fit_data, model_type='dependent')
    
    # 4. Generate predictions - Independent
    results_independent = predict_new_species(
        [model_independent_1, model_independent_2], 
        fit_data, 
        m_values, 
        model_type='independent'
    )
    
    # 5. Generate predictions - Dependent
    results_dependent = predict_new_species(
        model_dependent, 
        fit_data, 
        m_values, 
        model_type='dependent'
    )
    
    # 6. Compare models
    comparison = compare_models(results_independent, results_dependent, m_values)
    
    # 7. Create visualizations
    create_visualizations(results_independent, results_dependent, output_dir)
    
    # 8. Create output tables
    create_output_tables(results_independent, results_dependent, output_dir)
    
    # 9. Save results
    save_results(...)
    
    return {
        'independent': results_independent,
        'dependent': results_dependent,
        'comparison': comparison
    }
```

---

### **8. Utilities Module**

**8.1 Species Discovery Tracking**
```python
def track_new_species(predicted_samples, fit_samples_1, fit_samples_2):
    """Track which predicted species are new vs. shared."""
    # Returns: L^(0,0), L^(0,1), L^(1,0) counts
```

**8.2 HPD Interval Calculation**
```python
def compute_hpd_interval(samples, alpha=0.05):
    """Compute highest posterior density interval."""
```

**8.3 Linearity Check**
```python
def check_linearity(L_estimates, m_values):
    """Verify Theorem 1: L̂_m should be linear in m."""
    # Fit linear regression, return R^2 and slope
```

---

### **9. Configuration and CLI**

```python
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HPYP Species Sampling Experiment')
    parser.add_argument('--data', required=True, help='Path to input data')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--burn-in', type=int, default=5000)
    parser.add_argument('--output', default='./results')
    
    args = parser.parse_args()
    
    config = {
        'train_ratio': args.train_ratio,
        'num_fit_iterations': args.iterations,
        'burn_in': args.burn_in,
        'num_predict_iterations': args.iterations,
        'm_values': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'output_dir': args.output
    }
    
    results = run_experiment(args.data, config)
    print("Experiment completed successfully!")
```

---

### **10. Expected Outputs**

**Directory structure:**
```
results/
├── tables/
│   ├── independent_predictions.csv
│   ├── dependent_predictions.csv
│   ├── parameter_estimates.csv
│   └── model_comparison.csv
├── figures/
│   ├── fig1_independent.png
│   ├── fig2_dependent.png
│   ├── comparison_hpd_widths.png
│   └── parameter_posteriors.png
├── diagnostics/
│   ├── trace_plots.png
│   ├── convergence_stats.txt
│   └── linearity_check.txt
└── full_results.pkl  # Serialized Python object with all results
```

---

### **Key Implementation Notes**

1. **Use existing `pitmanyor.py`**: Extend the `HierarchicalPitmanYorProcess` class with:
   - Gibbs sampling for latent variables
   - Predictive distribution sampling
   - Parameter updates via Metropolis-Hastings

2. **Efficiency considerations**:
   - Vectorize probability calculations where possible
   - Use sparse data structures for large vocabularies
   - Implement progress bars for long-running samplers

3. **Reproducibility**:
   - Set random seeds
   - Save all configuration parameters
   - Version control for data and code

4. **Validation**:
   - Check parameter constraints (0 ≤ δ < 1, θ > -δ)
   - Verify probabilistic predictions sum to 1
   - Compare with paper results on EST data if available

This plan provides a complete blueprint for implementing the experimental methodology described in the paper, with clear separation of concerns and reusable modules.