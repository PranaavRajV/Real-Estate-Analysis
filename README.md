# 🏠 Real Estate Market Trends Predictor

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Updated](https://img.shields.io/badge/last%20updated-March%202024-emerald.svg)](#)

A production-grade end-to-end data science pipeline for predicting real estate market values and forecasting long-term trends. This project covers the full lifecycle from raw data synthesis to a live Power BI business intelligence dashboard.

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Pipeline Execution](#-pipeline-execution)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Project Overview
**The Business Problem:** Real estate investors and home buyers often lack transparent, data-driven tools to estimate property value and market trajectory. Basic regression models fail to account for spatial clustering and macroeconomic seasonality.

**The Solution:** This project implements:
1.  **Robust Cleaning:** Automated outlier detection (IQR + Z-Score).
2.  **Advanced Engineering:** Composite luxury scores and target-encoded spatial features.
3.  **ML Engine:** Comparative analysis of Gradient Boosting vs. Linear Ensembles.
4.  **Forecasting:** 12-month projections using Facebook Prophet.
5.  **BI Integration:** A 4-page Power BI dashboard for executive decision-making.

---

## 🏗 Architecture
```text
.
├── data/               # Raw, Processed, and Forecasted data
├── notebooks/          # 01_EDA -> 02_Modeling -> 03_Forecasting
├── scripts/            # Modular pipeline (Cleaning, Engineering, Training)
├── models/             # Serialized binaries (.pkl)
├── reports/            # Figures and Summary Reports
├── dashboards/         # Power BI .pbix files
├── config.yaml         # Centralized hyperparameters
└── requirements.txt    # Dependency tree
```

---

## 📊 Dataset
The project utilizes a synthesized dataset of **10,000+ properties** (or the Kaggle House Prices dataset) featuring:
- **Spatial:** Neighborhood type, Score, Location proxies.
- **Physical:** Sqft, Bedrooms, Bathrooms, Pool, Garage, Property Age.
- **Temporal:** Sale Date (2020-2024), Seasonal flags.

---

## ⚙️ Installation
1. **Clone the Repo:**
   ```bash
   git clone https://github.com/yourusername/real-estate-trends.git
   cd real-estate-trends
   ```
2. **Environment Setup:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 🚀 Pipeline Execution
Run the pipeline sequentially:
1. `python scripts/data_generator.py` (optional)
2. `python scripts/data_cleaning.py`
3. `python scripts/feature_engineering.py`
4. `python scripts/train_models.py`
5. `python scripts/forecasting.py`

---

## 🏆 Model Performance
The **Lasso Regressor** emerged as the primary valuation engine due to its high interpretability and strong linear alignment with market drivers.

| Model | R² Score | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| **Lasso Ensembles** | **0.9201** | **$36,895** | **$27,948** |
| XGBoost | 0.9148 | $38,101 | $29,038 |
| ARIMA (Forecast) | - | $12,266 | - |

---

## 💡 Key Insights
- **The Value of Space:** `sqft` explains ~70% of total valuation variance.
- **Amenity Premium:** A pool adds a median fixed value of **$25,300** across all types.
- **Trend Projection:** The market shows a stable **3% CAGR**, with peaks in Q2 (spring-buying cycle).

---

## ⚖️ Ethics & Responsible AI
As real estate data involves human habitats and financial futures, this project adheres to the following ethical considerations:
- **Bias Mitigation:** Location-based features (Neighborhood Score) can inadvertently reflect historical redlining or socio-economic biases. This model is intended for market analysis and should not be used as the sole basis for lending or insurance decisions.
- **Data Privacy:** All data used in this project is synthesized or anonymized. No Personal Identifiable Information (PII) of homeowners is stored or processed.
- **Fair Housing:** We acknowledge that automated valuation models (AVMs) can perpetuate systemic inequality. This project includes an accountability layer (SHAP) to ensure every price prediction is backed by specific, transparent physical attributes rather than "black-box" correlations.

---

## 🔮 Future Improvements
- [ ] Integration with Google Maps API for exact distance-to-city calculations.
- [ ] Real-time web scraper for Zillow/Redfin listings.
- [ ] Deployment of the pricing engine via a FastAPI endpoint or Streamlit app.

---

## 🚀 Deployment
This application is ready for deployment on **Streamlit Cloud**, **Heroku**, or **Render**.

### Deploying to Streamlit Cloud:
1. Push this repository to GitHub.
2. Sign in to [Streamlit Cloud](https://share.streamlit.io/).
3. Click "New App" and select this repository and `app/streamlit_app.py`.
4. The `requirements.txt` will automatically install all dependencies.

### Local Deployment:
```bash
streamlit run app/streamlit_app.py
```

---

## 👨‍💻 Author
**Pranaav Raj V**  
[LinkedIn](your-link) | [Portfolio](your-link) | [Email](your-email)
