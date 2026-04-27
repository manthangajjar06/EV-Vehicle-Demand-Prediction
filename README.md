# EV Adoption Forecaster — Washington State

> A machine learning web application that predicts county-level Electric Vehicle (EV) adoption trends across Washington State, forecasting cumulative EV counts up to 3 years into the future using a time-series-aware Random Forest model.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technical Architecture](#technical-architecture)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)

---

## Overview

As Electric Vehicle adoption accelerates globally, urban planners, infrastructure teams, and policymakers need reliable forecasts to anticipate charging station demand, grid load, and resource allocation. This project addresses that need by building a county-level EV adoption forecasting system for Washington State.

The application ingests historical EV registration data published by Washington State, engineers time-series features (lag values, rolling statistics, growth slopes), trains a Random Forest Regressor, and serves interactive 36-month forecasts through a Streamlit web interface. Users can explore forecasts for any individual county or compare trends across up to 3 counties simultaneously.

---

## Live Demo

> 🚀 [Launch App on Streamlit Cloud](https://your-app-url.streamlit.app) ← replace after deployment

---

## Features

- **County-level forecasting** — Select any county in Washington State and instantly view a 36-month (3-year) cumulative EV adoption forecast
- **Multi-county comparison** — Compare historical and projected EV adoption trends across up to 3 counties on a single interactive chart
- **Growth rate analysis** — Automatically computes and displays the expected percentage growth in EV adoption over the forecast period
- **Historical context** — Each forecast chart overlays real historical data alongside projections for full continuity
- **Responsive UI** — Clean dark-themed Streamlit interface optimized for readability

---

## Project Structure

```
EV-Vehicle-Demand-Prediction/
│
├── app.py                                      # Streamlit application — main entry point
├── EV_Adoption_Forecasting.ipynb               # End-to-end ML pipeline notebook
│
├── data/
│   ├── Electric_Vehicle_Population_By_County.csv   # Raw source dataset (WA state)
│   └── preprocessed_ev_data.csv                    # Feature-engineered dataset
│
├── model/
│   └── forecasting_tev_model.pkl               # Serialized trained Random Forest model
│
├── assets/
│   └── ev-car-factory.jpg                      # UI hero image
│
├── docs/
│   └── EV Forecast.pdf                         # Project report and analysis
│
├── requirements.txt                            # Pinned Python dependencies
└── README.md
```

> **Note:** The flat structure in the repository mirrors the above logical grouping. Paths in `app.py` reference files from the root directory.

---

## Technical Architecture

```
Raw Data (CSV)
     │
     ▼
┌─────────────────────────────────────────────┐
│           Data Preprocessing                │
│  • DateTime parsing & sorting               │
│  • Missing value removal                    │
│  • Outlier capping (IQR method)             │
│  • County label encoding                    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           Feature Engineering               │
│  • Lag features (lag-1, lag-2, lag-3)       │
│  • Rolling mean (3-month window)            │
│  • Percentage change (1-month, 3-month)     │
│  • EV growth slope (6-month linear fit)     │
│  • Months since first record (time index)   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│       Model Training & Evaluation           │
│  • Algorithm: Random Forest Regressor       │
│  • Tuning: RandomizedSearchCV               │
│  • Split: chronological 90/10               │
│  • Metrics: MAE, RMSE, R²                   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Recursive Forecasting Loop           │
│  • 36 iterations (1 month per step)         │
│  • Each prediction feeds next step's lags   │
│  • Cumulative EV count tracked per county   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
         Streamlit Web Application
```

---

## Dataset

| Property | Details |
|---|---|
| Source | [Washington State Open Data — EV Population by County](https://data.wa.gov) |
| Records | 20,819 rows |
| Features | 10 columns (Date, County, State, BEV count, PHEV count, EV Total, % EVs, etc.) |
| Coverage | Multiple counties across Washington State |
| Frequency | Monthly snapshots |
| Target variable | `Electric Vehicle (EV) Total` — monthly count per county |

**Key preprocessing decisions:**
- Outliers in `Percent Electric Vehicles` were capped to IQR bounds (rather than dropped) to preserve all county records while reducing extreme skew
- Counties with fewer than 3 historical records were excluded from forecasting (insufficient lag history)
- The dataset was sorted by `[County, Date]` before lag feature creation to prevent data leakage across county boundaries

---

## Model Details

### Algorithm — Random Forest Regressor

Random Forest was selected over ARIMA/SARIMA and gradient boosting methods for the following reasons:
- Handles non-stationary county growth curves without manual differencing
- Naturally captures non-linear interactions between lag features and time index
- Robust to the varying data density across counties (some counties have sparse records)
- Fast inference suits the real-time Streamlit interaction model

### Feature Set

| Feature | Description |
|---|---|
| `months_since_start` | Time index — monotonically increasing per county |
| `county_encoded` | Label-encoded county identifier |
| `ev_total_lag1` | EV count from 1 month prior |
| `ev_total_lag2` | EV count from 2 months prior |
| `ev_total_lag3` | EV count from 3 months prior |
| `ev_total_roll_mean_3` | 3-month rolling average |
| `ev_total_pct_change_1` | Month-over-month percentage change |
| `ev_total_pct_change_3` | 3-month percentage change |
| `ev_growth_slope` | Linear trend slope over last 6 months of cumulative EV count |

### Hyperparameter Tuning

Tuned via `RandomizedSearchCV` over:
- `n_estimators`: [100, 150, 200, 250]
- `max_depth`: [None, 5, 10, 15]
- Additional parameters from scikit-learn's RF implementation

### Forecasting Strategy

The model uses **recursive (multi-step) forecasting** — at each future time step, the model's own prediction is fed back as the lag input for the next step. This allows arbitrary-horizon forecasting without retraining, at the cost of compounding error over longer horizons.

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/EV-Vehicle-Demand-Prediction.git
cd EV-Vehicle-Demand-Prediction

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### Dependencies

```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
joblib==1.4.2
matplotlib==3.8.4
```

---

## Usage

**Single county forecast:**
1. Open the app
2. Use the **"Select a County"** dropdown to pick any WA county
3. The cumulative EV forecast chart renders immediately with historical data overlaid
4. A growth percentage summary appears below the chart

**Multi-county comparison:**
1. Scroll to the **"Compare EV Adoption Trends"** section
2. Use the multiselect dropdown to choose up to 3 counties
3. A combined chart renders all counties' historical + forecasted trends
4. Individual growth percentages are displayed for each selected county

---

## Results

The model demonstrates strong performance on the chronological test split (last 10% of records per county), with high R² scores across most counties. Counties with very sparse historical data naturally show higher uncertainty in long-range forecasts due to the recursive forecasting compounding effect.

Top counties by projected 3-year EV growth include high-population areas such as King, Pierce, and Snohomish counties, reflecting urbanization and existing EV infrastructure density.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web framework | Streamlit 1.35 |
| ML library | scikit-learn 1.4 |
| Data processing | pandas 2.2, NumPy 1.26 |
| Visualization | Matplotlib 3.8 |
| Model serialization | joblib 1.4 |
| Notebook environment | Jupyter |

---

## Future Improvements

- **Confidence intervals** — Add prediction intervals to forecast charts using quantile regression forests to communicate uncertainty more clearly
- **External features** — Incorporate gas price trends, EV incentive policy data, and charging station density as additional predictors
- **Model versioning** — Integrate MLflow or DVC to track experiments and model versions
- **Automated retraining** — Schedule monthly retraining via GitHub Actions as new WA state data is published
- **County-level model specialization** — Train separate models per county cluster (urban / suburban / rural) to reduce cross-county generalization error
- **SHAP explainability** — Add a model explanation panel showing which features drove each county's forecast

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Dataset provided by [Washington State Department of Licensing](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2)
- Built with [Streamlit](https://streamlit.io/) — open-source app framework for ML projects
