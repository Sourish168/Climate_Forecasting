# 🌦️ Weather Intelligence & Forecast App

An end-to-end Machine Learning web application built with Streamlit that predicts daily maximum and minimum temperatures. The app fetches real-time meteorological data and utilizes a hybrid time-series modeling architecture to generate a 7-day recursive forecast.

---

## 📖 Overview

This project demonstrates a complete machine learning lifecycle for time-series forecasting. It automatically ingests historical weather data and future Numerical Weather Prediction (NWP) forecasts, engineers temporal features, trains a hybrid statistical/tree-based model ensemble on the fly, and serves the results through an interactive dashboard.

## ✨ Key Features

*   **Automated Data Pipeline:** Dynamically fetches historical archive data and future NWP exogenous variables via the Open-Meteo API.
*   **Hybrid ML Architecture (SARIMA + XGBoost):** 
    *   Uses **SARIMA-X** as a base model to capture linear trends, weekly seasonality, and autoregressive properties.
    *   Uses **XGBoost** to model the non-linear residuals, utilizing an extensive engineered feature space.
*   **Recursive Forecasting:** Implements a multi-step forecasting loop where predictions are iteratively fed back into the lag features for future steps.
*   **Advanced Temporal Feature Engineering:** Automatically generates Fourier terms (sine/cosine for cyclicality), autoregressive lags, rolling window statistics, and time-step differencing.
*   **Interactive UI:** Built with Streamlit and Plotly for rich, interactive data visualization, including temperature bands, rain probabilities, and feature importance charts.

## 🛠️ Technology Stack

*   **Frontend / App Framework:** [Streamlit](https://streamlit.io/)
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** `xgboost`, `scikit-learn`
*   **Time Series Modeling:** `statsmodels` (SARIMAX)
*   **Data Visualization:** Plotly (`plotly.graph_objects`, `plotly.subplots`)
*   **APIs:** Government of India Website API, Requests

---

## 🚀 Installation & Setup

To run this project locally, ensure you have Python 3.8+ installed.

**1. Clone the repository**
```bash
git clone [https://github.com/Sourish168/Climate_Forecasting.git]
cd weather-forecast-app
```
---
## Use The Live Model Directly 

