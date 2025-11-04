# Air Quality ML Pipeline

A machine learning pipeline for air quality prediction using time-series data with geographic features.

## Overview

This project implements a complete, **CRISP-DM–aligned machine learning pipeline** for forecasting **PM2.5 air pollution levels** across several major African cities. The overarching goal is to transform raw environmental and temporal data into actionable insights that can inform **public-health policies and urban management strategies**.

The project’s objectives are twofold:

1. **Technical Objective:** Develop and evaluate multiple predictive models (Linear Regression, XGBoost, and LightGBM) capable of estimating PM2.5 concentrations with minimal bias and variance, while ensuring robustness across different geographic regions.

2. **Business Objective:** Provide decision-makers—such as municipal authorities and health agencies with **reliable short-term air-quality forecasts**. These forecasts enable timely public advisories (e.g., for schools, transport, outdoor activities) and help prioritize mitigation efforts like traffic management and street cleaning.

The work is grounded in the **Evaluation phase of the CRISP-DM framework**, where model performance is not only quantified through metrics such as RMSE and R² but also interpreted in terms of **real-world impact** and **policy relevance**. Rather than optimizing for accuracy alone, the emphasis is on **actionable reliability** whether predictions are consistent and informative enough to guide meaningful interventions.


## Dataset Description

The dataset used in this project was **provided by the course** and represents an integrated collection of environmental and atmospheric measurements across **four major African cities**. It forms the basis for developing and evaluating machine learning models that predict **PM2.5 concentration levels**, a key air-quality metric linked to respiratory health and urban livability.

### Source and Collection Period

The data covers the period **from January 1, 2023, to February 26, 2024**, capturing daily and hourly observations from monitoring sites located in:

* **Lagos (Nigeria)**
* **Nairobi (Kenya)**
* **Bujumbura (Burundi)**
* **Kampala (Uganda)**

### Dataset Structure

* **Training data shape (before cleaning):** 8,071 rows × 80 columns
* **Test data shape:** 2,783 rows × 79 columns
* **After cleaning:** 8,071 rows × **73 columns** (columns with >70% missing values dropped)
* **Geographic granularity:** city and site level (`city`, `country`, `site_id`, `site_latitude`, `site_longitude`)
* **Temporal granularity:** daily/hourly observations (`date`, `hour`)

### Dropped Columns

Columns removed due to excessive missing data (>70%):

```
['uvaerosollayerheight_aerosol_height',
 'uvaerosollayerheight_aerosol_pressure',
 'uvaerosollayerheight_aerosol_optical_depth',
 'uvaerosollayerheight_sensor_zenith_angle',
 'uvaerosollayerheight_sensor_azimuth_angle',
 'uvaerosollayerheight_solar_azimuth_angle',
 'uvaerosollayerheight_solar_zenith_angle']
```

### Missing Data Summary

After cleaning:

* **Train missing values:** 0
* **Test missing values:** 51,323 (handled by forward/backward imputation per city)

### Record Distribution by City

| City      | Number of Measurements |
| --------- | ---------------------- |
| Bujumbura | 123                    |
| Kampala   | 5,596                  |
| Lagos     | 852                    |
| Nairobi   | 1,500                  |

### Key Variables

| Variable                                          | Description                                                                               |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `pm2_5`                                           | Target variable — particulate matter ≤2.5µm concentration (µg/m³)                         |
| `city`, `country`                                 | Geographic identifiers used for GroupKFold cross-validation                               |
| `date`, `hour`                                    | Temporal identifiers for trend analysis and feature engineering                           |
| `site_id`, `site_latitude`, `site_longitude`      | Site-level geolocation metadata                                                           |
| `sulphurdioxide_so2_*`                            | Satellite-based SO₂ column density and correction parameters                              |
| `cloud_*`                                         | Cloud coverage, pressure, height, optical depth, albedo, and viewing angles               |
| (Other atmospheric and meteorological predictors) | Variables representing chemical and meteorological conditions influencing PM2.5 formation |

### Data Quality Notes

* **Column pruning:** Features with >70% missing data were dropped to prevent model bias and instability.
* **Imputation:** City-wise forward and backward fills were used to maintain temporal coherence.
* **No missing values remained** in the cleaned training data.
* **Class imbalance:** Not severe, though Kampala dominates with ~70% of records.
* **Outliers:** Retained to preserve the natural variability of PM2.5 levels.
* **Temporal consistency:** Continuous time series per city allow safe extraction of lagged and cyclical temporal features.

This dataset thus provides a **robust, multi-dimensional foundation** for evaluating how machine learning models generalize across distinct urban environments in sub-Saharan Africa.


## Project Structure

The repository follows a **modular and reproducible design**, aligning with the **CRISP-DM framework** and standard **MLOps** practices. Each component handles a specific phase of the machine learning workflow — from data processing to model evaluation and experiment tracking.

```
air_quality-main/
├─ src/
│  ├─ pipeline/
│  │  ├─ data_processor.py      # Handles data loading, missing-value treatment, and geographic GroupKFold creation
│  │  ├─ feature_engineer.py    # Generates temporal features and performs feature selection (SelectKBest, RFE)
│  │  ├─ model_trainer.py       # Defines ML models (Linear, Ridge, Lasso, XGBoost, LightGBM) and training logic
│  │  └─ evaluator.py           # Computes performance metrics, cross-validation, and MLflow logging
│  └─ utils/
│     ├─ config.py              # Centralized configuration (paths, thresholds, CV parameters, MLflow URI)
│     ├─ evaluation_utils.py    # Additional analysis tools (WHO threshold comparisons, visualization)
│     ├─ logger.py              # Logging utilities for structured and readable console output
│     └─ utils.py               # General-purpose helpers (plotting, file management)
│
├─ scripts/
│  ├─ run_pipeline.py           # End-to-end pipeline script (training, evaluation, MLflow integration)
│  └─ run_tests.py              # Unit test launcher with cryptographic proof generation
│
├─ tests/                       # Component-level tests ensuring reliability of preprocessing, features, and models
│
├─ data/                        # Folder for training and test CSVs (excluded from Git for privacy)
│
├─ mlruns/ and mlflow.db        # Local MLflow tracking storage (experiments, parameters, metrics, artifacts)
│
├─ mlartifacts/                 # Serialized trained models and associated metadata
│
└─ notebooks/                   # Optional exploratory notebooks used during development and validation
```

### Design Rationale

* **Separation of concerns:** Each module performs a single, well-defined role, simplifying debugging and future extensions.
* **Reproducibility:** The entire pipeline can be rerun from raw data to logged model using a single script (`run_pipeline.py`).
* **Scalability:** New models, features, or evaluation metrics can be added without modifying the overall structure.
* **Experiment traceability:** MLflow ensures every run is versioned and reproducible, supporting transparent model evaluation.

This structure supports both **technical robustness** and **business interpretability**, ensuring that every phase—from data preparation to evaluation—can be audited and communicated effectively to stakeholders.


## Installation

```bash
# Extract the project files
cd air_quality

# Install dependencies and package with uv
uv sync --extra dev

# Verify installation
uv run python scripts/run_tests.py --quick
```

## Usage

### Basic Pipeline

```bash
# Run basic pipeline
uv run python scripts/run_pipeline.py

# Different feature selection methods
uv run python scripts/run_pipeline.py --model linear --method rfe --n-features 15
```

### Advanced Models

[Add your advanced models usage here]

### MLflow Experiment Tracking

[Add your MLflow usage here]

## Key Findings

[Summarize your main analytical findings here:
- Key patterns discovered in the data
- Most important features for prediction
- Model performance comparisons
- Business insights and recommendations]

## Model Performance

[Document your model results:
- Performance metrics (RMSE, MAE, R²)
- Cross-validation results
- Feature importance rankings
- Comparison between different models]

## Methodology

[Describe your analytical approach:
- Data preprocessing steps
- Feature engineering strategy
- Model selection rationale
- Evaluation methodology]

## Authors

[Add team member names and task distribution:
- **Student 1 Name**: [Specific tasks and contributions]
- **Student 2 Name**: [Specific tasks and contributions]

Example:
- **Alice Dupont**: Data preprocessing, feature engineering, model evaluation
- **Bob Martin**: Model training, hyperparameter optimization, documentation]
