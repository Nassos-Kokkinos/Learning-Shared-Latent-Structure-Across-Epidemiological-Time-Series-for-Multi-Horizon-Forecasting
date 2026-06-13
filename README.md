# Learning Shared Latent Structure Across Epidemiological Time Series for Multi-Horizon Forecasting

This repository contains the codebase for a research project on **multi-series epidemiological forecasting**, with a focus on learning **shared latent structure** across related pathogen time series for **multi-horizon prediction**.

The project compares forecasting-specific neural baselines with a proposed shared-latent architecture under a **unified experimental pipeline**.

## Overview

The main goal of this work is to investigate whether related epidemiological time series can benefit from a forecasting framework that combines:

- **shared temporal representation learning**
- **series-aware conditioning**
- **pathogen-specific prediction heads**
- **multi-horizon forecasting**

The repository includes:

- data preprocessing and rolling-window construction
- baseline implementations of **N-BEATS** and **N-HiTS**
- the proposed **latent_shared** and **latent_shared_v2** models
- training scripts
- evaluation utilities
- forecast export pipelines
- generation of paper-ready tables and figures

---

## Forecasting Setting

The experiments are conducted in a **multi-series forecasting** setting over 6 pathogen-level time series:

- Adenovirus
- HCOV
- HMPV
- PIV
- RSV
- RV/EV

Each supervised sample is constructed using:

- **input window**: `L = 24`
- **forecast horizon**: `H = 4`

That is, each model receives the last 24 observations of a given series and predicts the next 4 values.

The temporal split is chronological:

- **train**
- **validation**
- **test**

This ensures an out-of-sample forecasting protocol without leakage from future observations.

---

## Repository Structure

```text
epi-forecasting-research/
├── configs/                     # YAML configuration files
├── data/
│   ├── raw/                     # Raw input data
│   └── processed/               # Processed/model-ready datasets
├── results/
│   ├── figures/                 # Generated figures
│   ├── forecasts/               # Exported forecasts
│   ├── metrics/                 # Evaluation metrics
│   ├── models/                  # Saved trained models
│   └── tables/                  # Paper-ready tables
├── scripts/                     # Executable experiment scripts
├── src/
│   └── epi_forecasting/
│       ├── data/                # Preprocessing and dataset construction
│       ├── evaluation/          # Metrics and evaluation helpers
│       ├── models/              # Forecasting model implementations
│       └── training/            # Training utilities
└── README.md