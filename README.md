# Bayesian Multi-Sample Species Sampling

A Python implementation of Hierarchical Pitman-Yor Processes (HPYP) for Bayesian multi-sample species sampling problems. This repository provides tools for fitting HPYP models, making predictions about novel species occurrence, and comparing independent versus dependent sampling strategies.

## Overview 🗺️

This implementation demonstrates the "borrowing of strength" phenomenon in Bayesian species sampling, where sharing information across multiple samples leads to more precise predictions. The code fits HPYP models using Gibbs sampling and generates predictions with uncertainty quantification.

## Repository Structure 📁

```
.
├── data/                          # Processed datasets ready for analysis
│   ├── namesbystate/              # Baby names by US state
│   ├── twenty+newsgroups/         # Newsgroup word frequencies
│   └── wilderness/                # Wilderness species data
│
├── data_raw/                      # Raw data files and preprocessing artifacts
│   ├── namesbystate/
│   ├── twenty+newsgroups/
│   └── wilderness/
│
├── scripts/                       # Core implementation and experiments
│   ├── data_utils.py              # Data loading and preprocessing
│   ├── pitmanyor.py               # HPYP model with Gibbs sampling
│   ├── model_fitting.py           # Independent and dependent model fitting
│   ├── prediction.py              # Species prediction algorithms
│   ├── output_utils.py            # Result formatting and table generation
│   ├── experiment.py              # Main experiment pipeline
│   ├── priors.py                  # Prior specification utilities
│   ├── config_*.json              # Experiment configurations
│   ├── run_experiments.sh         # Batch experiment runner
│   └── results_*/                 # Experiment outputs with tables and diagnostics
│
├── notes/                         # Documentation
│   ├── QUICKSTART.md              # Getting started guide
│   ├── IMPLEMENTATION_SUMMARY.md  # Technical implementation details
│   ├── IMPLEMENTATION_NOTES.md    # Development notes
│   ├── PRIORS_DOCUMENTATION.md    # Prior specification guide
│   ├── USAGE.md                   # Detailed usage instructions
│   ├── HPC_RUN_GUIDE.md           # High-performance computing guide
│   └── experiment_plan.md         # Experimental design documentation
│
├── playground.ipynb               # Interactive exploration notebook
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Quick Start (TODO) 🛠️


## Key Features 🎨

- **Independent Models**: Fit separate HPYP models for each sample
- **Dependent Models**: Share base distribution across samples for borrowing of strength
- **Gibbs Sampling**: Efficient posterior inference with parameter updates
- **Prediction**: Track multiple types of novel species occurrences
- **Uncertainty Quantification**: Compute posterior means and 95% HPD intervals
- **Model Comparison**: Quantify the benefit of sharing information

## Datasets 💽

The repository includes three example datasets:

1. **Baby Names by State**: Names given to babies in US states (aggregated by year and sex)
2. **20 Newsgroups**: Word frequencies across newsgroup categories (nouns, adjectives, verbs)
3. **Wilderness Data**: Species observations in wilderness areas

Add links to datasets (TODO).

Each dataset demonstrates different aspects of multi-sample species sampling.

## Documentation 📑

- See `notes/QUICKSTART.md` for a step-by-step tutorial
- See `notes/IMPLEMENTATION_SUMMARY.md` for technical details
- See `notes/USAGE.md` for command-line options and configuration