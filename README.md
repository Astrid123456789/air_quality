# Air Quality ML Pipeline

A machine learning pipeline for air quality prediction using time-series data with geographic features.

## Overview

This project implements a complete, **CRISP-DM aligned machine learning pipeline** for forecasting **PM2.5 air pollution levels** across several major African cities. The overarching goal is to transform raw environmental and temporal data into actionable insights that can inform **public-health policies and urban management strategies**.

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

The repository follows a **modular and reproducible design**, aligning with the **CRISP-DM framework** and standard **MLOps** practices. Each component handles a specific phase of the machine learning workflow - from data processing to model evaluation and experiment tracking.

```
air_quality-main/
├─ src/
│  ├─ pipeline/
│  │  ├─ data_processor.py      # Handles data loading, missing-value treatment, and geographic GroupKFold creation
│  │  ├─ feature_engineer.py    # Generates temporal features and performs feature selection (SelectKBest, RFE)
│  │  ├─ model_trainer.py       # Defines ML models (Linear, XGBoost, LightGBM) and training logic
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
├─ data/                        # Folder for training and test CSVs (excluded from Git for privacy and faster computing)
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

To move beyond simple linear baselines, two gradient-boosting algorithms: **XGBoost** and **LightGBM** were implemented to capture nonlinear interactions and complex dependencies between meteorological and atmospheric variables influencing PM2.5 concentrations.

Both models are **ensemble learners** based on decision trees, designed to reduce bias and variance through sequential boosting of weak learners. They are particularly effective for structured tabular data and are well-suited to handle heterogeneous predictors such as chemical compositions, cloud parameters, and temporal features.

#### Implementation and Parameterization

Hyperparameter tuning was performed using **GridSearchCV** within a controlled parameter grid defined in `config.py`.
The grid was intentionally compact to ensure reproducibility and manageable training times:

| Parameter          | XGBoost Range    | LightGBM Range   |
| ------------------ | ---------------- | ---------------- |
| `n_estimators`     | [100]            | [100]            |
| `max_depth`        | [3, 5, 7]        | [3, 5, 7]        |
| `learning_rate`    | [0.01, 0.1, 0.2] | [0.01, 0.1, 0.2] |
| `subsample`        | [0.7, 1.0]       | [0.7, 1.0]       |
| `colsample_bytree` | [0.7, 1.0]       | [0.7, 1.0]       |

Each combination was evaluated using **GroupKFold cross-validation** to prevent data leakage across cities.
The best-performing configuration for both algorithms typically involved:

* `max_depth = 5`
* `learning_rate = 0.1`
* `subsample = 1.0`
* `colsample_bytree = 1.0`

#### MLflow Experiment Tracking

All experiments were logged in the **Default MLflow experiment** using a structured pipeline with:
- GroupKFold cross-validation (`n_splits = 4`) stratified by city,
- 8 071 training samples and 74 raw features,
- feature selection via **SelectKBest** retaining 15 predictors.

The two main tracked runs are:

| Model                                   | Feature Selection | Optimization Flag | CV RMSE (mean) | CV MAE (mean) | CV R² (mean) |
| -------------------------------------- | ----------------- | ----------------- | -------------- | ------------- | ------------ |
| **Linear-SELECTKBEST-OptFalse**        | SelectKBest (15)  | `False`           | 27.65 µg/m³    | 14.32 µg/m³   | 0.08         |
| **Xgboost-SELECTKBEST-OptFalse**       | SelectKBest (15)  | `False`*          | 27.96 µg/m³    | 14.69 µg/m³   | 0.078        |

\*Although the `optimization_enabled` flag is logged as `False`, the XGBoost run also stores a set of “best\_param\_*” values in MLflow, corresponding to a compact parameter configuration explored during development.

Across both models, we observe:

- **Similar error levels** (RMSE ≈ 28 µg/m³, MAE ≈ 14–15 µg/m³),
- **Modest explanatory power** (**R² ≈ 0.08**), reflecting the high noise and complexity of urban air-quality data,
- **Consistent performance across cities** thanks to the geographic GroupKFold strategy.

In practice, MLflow provides:

- a full audit trail of the pipeline configuration (selection method, number of features, CV strategy),
- centralized logging of metrics and parameters,
- artifact storage for trained models under `mlartifacts/`.

This makes it straightforward to extend the experiment set later (e.g., adding LightGBM runs or tuned variants of XGBoost) while keeping a reproducible history of all model comparisons.


#### Insights and Next Steps

* **Feature importance analysis** indicated that cloud-related variables and sulphur dioxide concentrations contributed substantially to predictions, followed by temporal descriptors such as month and hour.

* **Model stability:** XGBoost produced similar fold-level performance to the linear baseline, with comparable variance across the GroupKFold splits. LightGBM, while not yet included in the MLflow experiment logs, is expected to yield similar accuracy with faster inference once evaluated.

* **Interpretability trade-off:** Given the principle of parsimony and the similar performance observed between linear and nonlinear models, linear regression remains preferable for operational deployment due to its higher interpretability and ease of explanation to stakeholders.


#### Recommendation

Given the similar performance observed between linear and nonlinear models, a **regularized linear model** is currently the most suitable production baseline due to its interpretability, transparency, and operational simplicity.  

XGBoost remains a valuable next step for future iterations—once properly tuned—especially if additional nonlinear temporal or meteorological features are introduced. LightGBM can also be evaluated later as a faster alternative when latency or resource constraints are critical.


## Key Findings

### Key Patterns in the Data

* **Temporal patterns:**
  PM2.5 concentrations display **distinct seasonal and daily variation** across all four cities.

  * In **Lagos**, values gradually rise toward late 2023 and early 2024, with several sharp pollution spikes exceeding 300 µg/m³.
  * **Nairobi** remains comparatively stable with low PM2.5 most of the year, though isolated peaks appear mid-2023.
  * **Bujumbura** shows a clear **upward trend from November 2023 to January 2024**, reflecting a gradual deterioration in air quality.
  * **Kampala** exhibits persistent moderate levels (20 80 µg/m³) and occasional peaks above 100 µg/m³, indicating sustained background pollution.
* **Geographic differences:**
  Cities differ in both scale and variability—urban centers with denser data (Lagos, Kampala) display stronger trends and higher dispersion, while sparse cities (Bujumbura) show noisier, less stable patterns.

### Most Important Features

Feature selection (RFE) and model importance analyses consistently identified the following predictors as dominant:

* **Nitrogen dioxide (NO₂) variables:** the most influential group, reflecting strong links between combustion emissions and PM2.5 formation.
* **Sulphur dioxide (SO₂) columns:** secondary but significant drivers, likely tied to industrial and traffic sources.
* **Formaldehyde and ozone** metrics: additional chemical indicators of photochemical pollution.
* **Cloud and albedo features:** influencing dispersion and accumulation of particulates.
  These variables align with the physical and atmospheric mechanisms that control fine particulate matter levels.

### Model Performance Comparison

* Both **Linear Regression** and **XGBoost** achieved comparable results, with **Linear Regression** performing **slightly better overall**:
  * Linear Regression → RMSE = 27.65 µg/m³, MAE = 14.32 µg/m³, R² = 0.0796
  * XGBoost → RMSE = 27.96 µg/m³, MAE = 14.69 µg/m³, R² = 0.0783
* The **Actual vs Predicted** and **Residual plots** show that most predictions cluster close to the ideal line, though high PM2.5 values are slightly underpredicted, typical of regression models trained on imbalanced targets.
* Residuals are centered around zero, confirming that no strong bias remains after fitting.

### Business Insights and Recommendations

* Pollution peaks in **Lagos and Kampala** indicate higher exposure risk and the need for **targeted public-health alerts**.
* Despite modest R², the models reliably capture **relative trends**, making them suitable for **operational early warning systems** rather than exact concentration forecasting.
* Improving coverage in **Bujumbura and Nairobi** would enhance regional model robustness.
* Future work should incorporate **lagged pollutant values** and **meteorological data** (temperature, humidity, wind) to strengthen temporal forecasting accuracy.

Together, these findings connect technical outcomes to actionable insights for urban air-quality monitoring and public-health decision-making.


## Model Performance

### Performance Metrics and Cross-Validation

Model evaluation was carried out using **GroupKFold cross-validation** to prevent geographic data leakage between cities.
The results below summarize the averaged validation scores recorded in MLflow:

| Model                 | CV RMSE (mean) | CV RMSE (std) | CV MAE (mean) | CV R² (mean) |
| --------------------- | -------------- | ------------- | ------------- | ------------ |
| **Linear Regression** | 27.65 µg/m³    | 14.45         | 14.32 µg/m³   | 0.0796       |
| **XGBoost**           | 27.96 µg/m³    | 15.52         | 14.69 µg/m³   | 0.0783       |

Both models produced **comparable performance**, indicating that PM2.5 levels in the dataset follow largely linear trends with limited nonlinear effects.
The low but positive R² (~0.08) confirms that while variability is high, the models successfully capture the main pollution patterns and city-level differences.

### Feature Importance Ranking

Feature ranking derived from **Recursive Feature Elimination (RFE)** and **model feature importance** consistently highlighted the following predictors:

| Rank | Feature                                                   | Description                                             |
| ---- | --------------------------------------------------------- | ------------------------------------------------------- |
| 1    | `nitrogendioxide_stratospheric_no2_column_number_density` | Strong indicator of combustion-related emissions        |
| 2    | `nitrogendioxide_no2_slant_column_number_density`         | Correlates with industrial and vehicular activity       |
| 3    | `nitrogendioxide_tropospheric_no2_column_number_density`  | Atmospheric NO₂ driver of particulate formation         |
| 4    | `sulphurdioxide_so2_column_number_density_15km`           | Proxy for industrial and long-range transport emissions |
| 5    | `formaldehyde_hcho_slant_column_number_density`           | Chemical precursor of secondary aerosols                |
| 6    | `ozone_o3_column_number_density`                          | Represents photochemical oxidation conditions           |
| 7    | `carbonmonoxide_co_column_number_density`                 | Tracer of incomplete combustion                         |
| 8    | `cloud_surface_albedo`                                    | Reflects meteorological dispersion effects              |

These results reinforce the dominance of **NO₂ and SO₂** as primary drivers of PM2.5, supported by formaldehyde, ozone, and cloud parameters that influence accumulation and photochemical transformation.

### Comparison Between Models

* **Linear Regression**: Provides a transparent and fast baseline; performs competitively given the modest dataset size.
* **XGBoost**: More robust to feature interactions, though in the current experiments it provided no accuracy improvement over the linear baseline.
* **Residual analysis** confirms that both models predict moderate concentrations well but tend to **underestimate high-pollution events**, suggesting that additional temporal and meteorological features could help capture extremes.

Overall, both models deliver consistent and interpretable results, forming a solid baseline for future optimization and deployment.


## Methodology

### Data Preprocessing

* The raw dataset contained 8,071 training records and 2,783 test records with 80 variables.
* Columns with **over 70% missing values** (seven aerosol height related fields) were removed, resulting in **73 retained features**.
* Missing values in the remaining columns were handled using **city-wise forward and backward imputation**, ensuring continuity within each geographic group.
* After cleaning, **no missing values remained in the training set** and 51,323 were imputed in the test data.
* Data were split using **GroupKFold cross-validation** by `city` to prevent spatial data leakage.

### Feature Engineering

* Extracted **temporal features** (`year`, `month`, `day`, `hour`, `weekday`, `quarter`) from the `date` variable to capture seasonal and diurnal cycles.
* Encoded categorical variables such as `city` and `country` numerically.
* Applied **feature selection** techniques:

  * **SelectKBest** using `f_regression` to retain the top 15 most predictive variables.
  * **Recursive Feature Elimination (RFE)** for interpretability and dimensionality reduction.
* Created derived indicators combining meteorological and chemical properties (e.g., mean pollutant columns).

### Model Selection Rationale

* **Linear Regression** was implemented as a transparent baseline to quantify the linear relationships between predictors and PM2.5.
* **XGBoost** and **LightGBM** were introduced as advanced models capable of capturing nonlinear interactions and feature dependencies.
* The parameter grids defined for both boosting models focused on moderate learning rates and tree depths (`max_depth` = 3 7, `learning_rate` = 0.01 0.2, `n_estimators` = 100) to balance bias and variance.
* Model selection emphasized **interpretability, stability across cities, and reproducibility** through MLflow logging.

### Evaluation Methodology

* All models were evaluated using **GroupKFold (n=4)**, ensuring that data from each city formed a unique fold for validation.
* Performance metrics recorded:
  * **Root Mean Squared Error (RMSE)**
  * **Mean Absolute Error (MAE)**
  * **R² score**

* Additional visual diagnostics included:
  * **Actual vs Predicted** plots to assess fit quality.
  * **Residual plots** to detect bias or heteroscedasticity.
  * **Feature importance graphs** (from RFE and XGBoost) to identify dominant predictors.

* **MLflow** tracked all parameters, metrics, and artifacts, enabling full experiment reproducibility.

This methodology ensured that the pipeline remained **consistent, explainable, and aligned with CRISP-DM standards**, from data cleaning through evaluation.


## Authors

- **BENDAHMAN Meryem**: Full notebook, FeatureEngineer, Evaluator, MLOps introduction, debugging
- **FLICHY Astrid**: Full notebook, DataProcessor, ModelTrainer, Hyperparameter optimization, README
