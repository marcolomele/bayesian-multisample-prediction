# Quick Start Guide: HPYP Multi-Sample Experiment

## Installation

No additional dependencies needed beyond what's already installed:
- pandas
- numpy
- scipy
- tqdm

## Running Your First Experiment

### Step 1: Validate Installation

```bash
cd scripts
python3 test_pipeline.py
```

Expected output:
```
✓ Data loading tests passed!
✓ Model creation tests passed!
✓ Model fitting tests passed!
✓ Prediction tests passed!
✓ ALL TESTS PASSED!
```

### Step 2: Quick Test Run

```bash
cd scripts
python3 experiment.py --quick-test --data ../data/namesbystate_test.csv
```

This runs with:
- 100 Gibbs iterations (50 burn-in)
- 100 prediction iterations
- 2 m values: [200, 400]
- Small test dataset (5,000 rows)

**Runtime**: ~10-20 minutes

### Step 3: View Results

```bash
ls -R results_quick_test/
cat results_quick_test/tables/independent_predictions.csv
cat results_quick_test/tables/dependent_predictions.csv
cat results_quick_test/tables/model_comparison.csv
```

### Step 4: Full Experiment

```bash
python3 experiment.py --data ../data/namesbystate_subset.csv --output ../results_full
```

This runs with:
- 1,000 Gibbs iterations (500 burn-in)
- 1,000 prediction iterations  
- 10 m values: [200, 400, ..., 2000]
- Full dataset (~1.5M rows)

**Runtime**: ~2-3 hours (depends on hardware)

## Understanding the Output

### Independent Model Results
Shows predictions without borrowing strength across states:

| m | CA_L0 | CA_HPD | FL_L0 | FL_HPD | ... |
|---|-------|---------|-------|---------|-----|
| 200 | 68.21 | (54, 83) | ... | ... | ... |

- `CA_L0`: Expected number of new names in California
- `CA_HPD`: 95% credible interval

### Dependent Model Results
Shows predictions with borrowing strength:

| m | state | L_0_0 | L_from_TX | L_from_NY | ... | L_0 | HPD |
|---|-------|-------|-----------|-----------|-----|-----|-----|
| 200 | CA | 45.2 | 8.3 | 12.5 | ... | 67.5 | (55, 81) |

- `L_0_0`: New names not seen in ANY state
- `L_from_TX`: Names in TX but new to CA
- `L_0`: Total new names to CA
- `HPD`: Narrower interval (borrowing of strength effect!)

### Model Comparison
Shows the benefit of borrowing strength:

| m | state | independent_mean | dependent_mean | independent_hpd_width | dependent_hpd_width | width_reduction_pct |
|---|-------|------------------|----------------|----------------------|---------------------|---------------------|
| 1000 | CA | 340.37 | 266.13 | 88 | 67 | 23.9% |

**Key insight**: Dependent model has ~24% narrower credible intervals!

## Customization

### Custom Configuration File

Create `my_config.json`:
```json
{
  "train_ratio": 0.8,
  "seed": 42,
  "states": ["CA", "TX", "NY"],
  "num_fit_iterations": 2000,
  "burn_in": 1000,
  "num_predict_iterations": 2000,
  "m_values": [100, 500, 1000, 2000],
  "d_0": 0.5,
  "theta_0": 1000.0,
  "d_j": 0.5,
  "theta_j": 1000.0,
  "update_params": true,
  "verbose": true,
  "output_dir": "./my_results"
}
```

Run with:
```bash
python3 experiment.py --config my_config.json --data ../data/namesbystate_subset.csv
```

### Command-Line Options

```bash
# Specify output directory
python3 experiment.py --output ./my_results

# Use different data file
python3 experiment.py --data /path/to/my/data.csv

# Combine options
python3 experiment.py --config my_config.json --data my_data.csv --output ./results
```

## Interpreting Results

### Parameter Estimates Table

```
| model | state | theta_0 | d_0 | theta_j | d_j |
|-------|-------|---------|-----|---------|-----|
| Independent | CA | 1213.4 ± 45.2 | 0.4676 ± 0.02 | ... | ... |
| Dependent | ALL | 1044.5 ± 38.1 | 0.3449 ± 0.01 | ... | ... |
```

- **θ_0, θ_j**: Concentration parameters (higher = more new species)
- **δ_0, δ_j**: Discount parameters (0 ≤ δ < 1, controls clustering)
- Dependent model shares parameters across all states

### Linearity Check

File: `diagnostics/linearity_dependent.txt`

```
Group 0 (CA):
  Slope: 0.3381
  R²: 0.999823
  ✓ Excellent linear fit (R² > 0.99)
```

Verifies Theorem 1 from the paper: E[L_m] is linear in m.

### Borrowing of Strength

The key result comparing independent vs dependent models:

1. **Narrower HPD intervals** in dependent model
   - Less uncertainty about predictions
   - More efficient use of data

2. **Similar or better point estimates**
   - Dependent model pools information
   - Especially benefits states with less data

3. **Shared species discovery**
   - Tracks which species are shared across states
   - Identifies truly novel vs. state-specific species

## Troubleshooting

### Out of Memory
- Use smaller dataset (test file)
- Reduce iterations
- Process fewer states at once

### Slow Performance
- Use `--quick-test` flag
- Reduce `m_values` list
- Disable parameter updates: `"update_params": false`

### Unexpected Results
1. Check data format matches expected structure
2. Verify random seed for reproducibility
3. Increase iterations if results seem unstable
4. Review diagnostics/linearity files

## Next Steps

After getting familiar with the basic pipeline:

1. **Experiment with different states**
   - Try different combinations
   - Compare coastal vs. inland states
   - Analyze temporal trends (if keeping years separate)

2. **Vary hyperparameters**
   - Test sensitivity to θ and δ
   - Compare HDP (δ=0) vs HPYP

3. **Add visualizations** (Phase 2)
   - Plot L^0_m vs m for each state
   - Compare HPD intervals visually
   - Show parameter posteriors

4. **Different datasets**
   - Try with microbiome data
   - Test on EST data (like the paper)
   - Apply to your own species sampling problems

## Example Analysis Workflow

```bash
# 1. Quick validation
python3 test_pipeline.py

# 2. Small test to check setup
python3 experiment.py --quick-test --data ../data/namesbystate_test.csv

# 3. Review quick results
head results_quick_test/tables/*.csv

# 4. If satisfied, run full experiment
python3 experiment.py --data ../data/namesbystate_subset.csv

# 5. Analyze results
python3 -c "
import pandas as pd
comp = pd.read_csv('results/tables/model_comparison.csv')
print(comp[comp['m'] == 1000])
"

# 6. Check linearity
cat results/diagnostics/linearity_dependent.txt
```

## Questions?

- Check `IMPLEMENTATION_SUMMARY.md` for technical details
- Review `experiment_plan.md` for original design
- See paper_v2.pdf for theoretical background
- Run `test_pipeline.py` to diagnose issues

---

**Ready to begin!** Start with Step 1 above.

