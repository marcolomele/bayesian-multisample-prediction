# Chinese Restaurant Franchise (CRF) Terminology

The pipeline now uses standard CRF terminology throughout, making it applicable to any species sampling problem.

## Terminology Mapping

| CRF Term | Meaning | Examples |
|----------|---------|----------|
| **Restaurant** | Group/Sample | State, population, library, corpus |
| **Dish** | Species/Atom | Name, gene, word, species |
| **Customer** | Observation | Person, sequence, document, individual |
| **Table** | Cluster | Group of customers at same table serving same dish |

## Generic Data Format

Your data should be in one of these formats:

### Format 1: CRF Format (Recommended)
```csv
restaurant,dish,count
CA,John,1500
CA,Mary,1200
TX,John,1800
TX,Mary,1000
```

### Format 2: Custom Column Names
```csv
state,name,total_count
CA,John,1500
CA,Mary,1200
```

The pipeline auto-detects column names and converts to CRF format internally.

## Generic Data Loading

### Using `load_species_data()`

```python
from data_utils import load_species_data

# Generic loader - works with any species sampling data
df = load_species_data(
    data='my_data.csv',
    restaurant_col='state',      # Column for restaurants
    dish_col='name',              # Column for dishes
    count_col='count',            # Column for customer counts
    max_customers_per_restaurant=100000,  # Optional cap
    restaurants=['CA', 'TX'],     # Optional filter
    random_seed=42
)
```

### Example: EST Data (like the paper)
```python
# EST libraries as restaurants, genes as dishes
df = load_species_data(
    data='est_data.csv',
    restaurant_col='library',
    dish_col='gene',
    count_col='expression_count',
    restaurants=['FRUIT1', 'FRUIT2']
)
```

### Example: Microbiome Data
```python
# Samples as restaurants, species as dishes
df = load_species_data(
    data='microbiome.csv',
    restaurant_col='sample_id',
    dish_col='species',
    count_col='abundance'
)
```

## Data Processing Functions

### Split Customers
```python
from data_utils import split_customers_random

fit_data, predict_data = split_customers_random(
    df,
    restaurant_col='restaurant',
    dish_col='dish',
    count_col='total_count',
    train_ratio=0.8,
    random_seed=42
)
```

### Prepare CRF Input
```python
from data_utils import prepare_crf_input

data_dict, metadata = prepare_crf_input(
    df,
    restaurants=['CA', 'TX'],
    restaurant_col='restaurant',
    dish_col='dish',
    count_col='count'
)

# Returns:
# - data_dict: {restaurant_id: [(dish, customer_count), ...]}
# - metadata: {
#     'restaurants': ['CA', 'TX'],
#     'restaurant_to_id': {'CA': 0, 'TX': 1},
#     'id_to_restaurant': {0: 'CA', 1: 'TX'},
#     'n_i': {0: 100000, 1: 100000},  # customers per restaurant
#     'K_n_i': {0: 5000, 1: 6000},    # unique dishes per restaurant
#     'all_dishes': set(...),          # all unique dishes
#     'num_restaurants': 2
#   }
```

### Expand Customers
```python
from data_utils import expand_customers

# Convert (dish, count) to list of dishes
expanded = expand_customers(data_dict)
# Returns: {restaurant_id: [dish, dish, dish, ...]}
```

## Legacy Functions (Backward Compatible)

The following functions still work but use legacy terminology:

- `load_names_data()` - Loads baby names data
- `load_names_data_stratified()` - Loads with stratified sampling
- `split_data_random()` - Auto-detects columns and splits
- `prepare_hpyp_input()` - Auto-detects columns and prepares input
- `expand_observations()` - Alias for `expand_customers()`

## Example: Complete Workflow

```python
from data_utils import (
    load_species_data,
    split_customers_random,
    prepare_crf_input,
    expand_customers
)

# 1. Load data
df = load_species_data(
    'my_data.csv',
    restaurant_col='sample',
    dish_col='species',
    count_col='count',
    max_customers_per_restaurant=100000
)

# 2. Split into fit/predict
fit_data, predict_data = split_customers_random(
    df,
    restaurant_col='restaurant',
    dish_col='dish',
    count_col='total_count',
    train_ratio=0.8
)

# 3. Prepare for HPYP
restaurants = ['CA', 'TX']
data_dict, metadata = prepare_crf_input(
    fit_data,
    restaurants=restaurants
)

# 4. Expand for fitting
expanded_data = expand_customers(data_dict)
# Now ready for HPYP fitting!
```

## Benefits of CRF Terminology

1. **Generic**: Works with any species sampling problem
2. **Standard**: Uses established Bayesian nonparametrics terminology
3. **Clear**: Intuitive metaphor (restaurants, dishes, customers)
4. **Flexible**: Auto-detects column names for backward compatibility

## Migration Guide

If you have existing code using `state`/`name` terminology:

1. **Option 1**: Keep using legacy functions (they auto-detect)
2. **Option 2**: Convert to CRF format:
   ```python
   df_crf = convert_names_to_crf(df, state_col='state', name_col='name')
   ```
3. **Option 3**: Use generic functions with column mapping:
   ```python
   df = load_species_data(
       'data.csv',
       restaurant_col='state',
       dish_col='name',
       count_col='count'
   )
   ```

All approaches work seamlessly!

