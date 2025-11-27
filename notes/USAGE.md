# HPYP Experiment Pipeline - Usage

## Quick Start

Run an experiment with a configuration file:

```bash
python experiment.py config_news.json
```

Resume from checkpoint if interrupted:

```bash
python experiment.py config_news.json --resume
```

## Configuration File

The config file specifies all experiment parameters. Required fields:

```json
{
  "data_path": "../data/newsgroup_words.csv",
  "restaurant_col": "category",
  "dish_col": "word", 
  "count_col": "count",
  "restaurants": ["sci.space", "soc.religion.christian"],
  "max_customers_per_restaurant": 20000,
  "train_ratio": 0.8,
  "seed": 42,
  "num_fit_iterations": 100,
  "burn_in": 50,
  "num_predict_iterations": 100,
  "m_values": [200, 500, 1000, 1500, 2000],
  "d_0": 0.5,
  "theta_0": 1000.0,
  "d_j": 0.5,
  "theta_j": 1000.0,
  "update_params": true,
  "verbose": true,
  "output_dir": "./results_news_quick_test"
}
```

### Key Parameters

- **data_path**: CSV file with (restaurant, dish, count) data
- **restaurant_col, dish_col, count_col**: Column names in CSV (if different from 'restaurant', 'dish', 'count')
- **restaurants**: List of restaurants to include (optional; uses all if omitted)
- **max_customers_per_restaurant**: Cap on samples per restaurant (optional)
- **train_ratio**: Train/test split ratio
- **seed**: Random seed for reproducibility
- **num_fit_iterations, burn_in**: MCMC fitting parameters
- **num_predict_iterations**: Prediction iterations
- **num_threads**: Number of threads for parallel sampling (default: 1)
- **m_values**: Sample sizes for prediction
- **d_0, theta_0, d_j, theta_j**: HPYP hyperparameters
- **output_dir**: Where to save results

## Performance: Multithreading

The pipeline supports multithreading to accelerate:
1. **Independent model fitting**: Each restaurant model fits in parallel
2. **Prediction sampling**: Prediction iterations run in parallel

Set `num_threads` in config to enable (e.g., `"num_threads": 4`):

```json
{
  "num_threads": 4,
  ...
}
```

**Note**: Speedup is approximately linear with number of threads for prediction iterations. Model fitting speedup depends on number of restaurants.

## Output Structure

Results are saved to `output_dir/`:

```
results_news_quick_test/
├── tables/
│   ├── independent_predictions.csv
│   ├── dependent_predictions.csv
│   ├── parameter_estimates.csv
│   └── model_comparison.csv
├── diagnostics/
│   ├── linearity_independent.txt
│   └── linearity_dependent.txt
├── checkpoints/
│   ├── models.pkl
│   ├── predictions.pkl
│   └── statistics.pkl
├── full_results.pkl
└── config.json
```

## Example Configs

Three example configs are provided:

- **config_news.json**: Newsgroup words experiment
- **config_names.json**: Baby names by state experiment
- **config_wilderness.json**: Wilderness soil types experiment

## Resume Feature

If an experiment is interrupted, use `--resume` to continue from the last checkpoint:

```bash
python experiment.py config.json --resume
```

Checkpoints are saved after:
1. Model fitting
2. Predictions
3. Statistics computation

## Data Format

Input CSV should have three columns (names configurable):
- **restaurant**: Group identifier (e.g., state, category)
- **dish**: Item identifier (e.g., name, word, species)
- **count**: Number of observations

Example:

```csv
restaurant,dish,count
sci.space,satellite,42
sci.space,orbit,38
soc.religion.christian,church,56
```

