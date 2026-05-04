# wilson-motor-insurance-project
Predictive Analytics for Motor Insurance Risk /  MSc Business Analytics and Technology, University of Greater Manchester
# Predictive Analytics for Motor Insurance Risk

**Author:** Agene Osemegbe Wilson
**Student ID:** 2434966
**Programme:** MSc Business Analytics and Technology
**Institution:** University of Greater Manchester
**Year:** 2026

## Project Overview

This repository contains the analytical code for my MSc dissertation:
*Predictive Analytics for Motor Insurance Risk: Time-Series Forecasting
of Road Collisions to Support Claims Prediction*

The study analyses 503,475 UK road collision records (2020–2024) using
time series forecasting and logistic regression to support motor
insurance decision-making.

## Repository Contents

| File | Description |
|------|-------------|
| `Wilson_Dashboard_Streamlit_v2.py` | Interactive decision-support dashboard built in Python using Streamlit |
| `Wilson_Analysis_Notebook.ipynb` | Full analysis notebook — data loading, EDA, forecasting models, regression |

## Key Findings

- **Best forecasting model:** Holt-Winters ETS (MAPE 3.41%)
- **Automated SARIMA selection failed** during COVID structural disruption — manual parameters recovered 12% accuracy improvement
- **Top severity risk factors:** Rural location (OR 1.138), Speed limit (OR 1.122), Darkness (OR 1.092), Weekend (OR 1.066)
- **Adverse weather** shows risk compensation effect — frequency risk, not severity risk

## Data Sources

- [DfT STATS19 Road Safety Data](https://www.data.gov.uk/dataset/road-accidents-safety-data) — Open Government Licence v3.0
- [DESNZ Weekly Road Fuel Prices](https://www.gov.uk/government/statistics/weekly-road-fuel-prices) — Open Government Licence v3.0

## How to Run the Dashboard

1. Install required libraries: pip install streamlit pandas numpy matplotlib statsmodels pmdarima scikit-learn
2. Run the dashboard:streamlit run wilson_project.py
3. 3. Upload the STATS19 collision CSV and the monthly time series CSV
   when prompted in the sidebar.

## Technologies Used

- Python 3.12
- Streamlit — dashboard framework
- pandas, numpy — data processing
- statsmodels, pmdarima — time series modelling
- scikit-learn — logistic regression
- matplotlib — visualisation
