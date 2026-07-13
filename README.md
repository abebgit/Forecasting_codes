# Deployment-Oriented, Explainable Day-ahead Wind Power Prediction at the Turbine Level

## Overview

This repository contains source code for data preprocessing, forecasting models, explainability analysis, and statistical evaluation used in the study:

"Deep Learning-based Day-ahead Wind Power Forecasting for Adama-II Wind Turbines and Conceptual Deployment Pathways."

## Repository Structure

- `preprocessing/` – Data cleaning, feature engineering, and Data splitting.
- `models/` – Stacked-LSTM, Wavelet-LSTM, CNN-LSTM, Wavelet-CNN and XGBoost
- `explainability/` – SHAP Analysis and Feature importance.
- `statistical_tests/` – Paired t-test and Bootstrap Confidence Intervals on the already predicted files.

## Requirements

The code was developed and tested in Python 3.12 using the following libraries:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- TensorFlow
- SHAP
- Psutil
- PyWavelets

Install all dependencies using:

```bash
pip install -r requirements.txt
