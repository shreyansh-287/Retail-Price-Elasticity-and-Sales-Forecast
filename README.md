# 🛒 Retail Weekly Sales Forecasting API

An end-to-end Retail Weekly Sales Forecasting project built using time-series feature engineering, Ridge Regression, and deployed as a public FastAPI service.

🔗 **Live API:**  
https://retail-forecast-api-vyta.onrender.com/docs

---

## 🚀 Project Overview

This project builds a production-ready forecasting pipeline for predicting weekly retail sales at the store level.

The system includes:

- Exploratory Data Analysis (EDA)
- Time-series feature engineering (lag & rolling features)
- Regularized Linear Model (Ridge Regression)
- Time-based cross-validation
- Hyperparameter tuning using GridSearchCV
- Model serialization using Pickle
- Deployment using FastAPI
- Public hosting on Render

---

## 📊 Problem Statement

Forecast weekly store-level retail sales using historical performance, promotional activity, and store metadata.

Key challenges addressed:

- Temporal dependence in retail sales
- Multicollinearity due to lag features and store identifiers
- Preventing data leakage in time-series modeling
- Building a reusable production pipeline

---

## 🧠 Feature Engineering

### Time-Series Features
- `lag_1` → Previous week's sales
- `lag_4` → Sales 4 weeks ago
- `rolling_mean_4` → Rolling average of last 4 weeks

### Store & Business Features
- `Promo`
- `CompetitionDistance`
- `SchoolHoliday`
- `StoreType`
- `Assortment`
- `Store` (store-level fixed effects)

Target Variable:
- `log_sales` (log-transformed weekly sales)

---

## 🏗️ Model Architecture

Pipeline built using:

- `ColumnTransformer`
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
- `Ridge Regression` (to control multicollinearity)

---

## 🔎 Validation Strategy

Time-series aware validation:

- Chronological train-test split
- `TimeSeriesSplit` (5-fold cross-validation)

### Final Performance

- **Holdout R²:** ~0.91
- **Cross-Validation R² (mean):** ~0.78
- **Best Alpha (Ridge):** 1.0

---

## 💾 Model Serialization

The complete preprocessing + modeling pipeline was serialized using: pickle

This ensures consistent transformations in both training and production.

## 🌐 API Deployment

The model is deployed using **FastAPI**.

### Endpoint: Prediction Endpoint (POST /predict)

### Sample Request

```json
{
  "Store": 1,
  "Promo": 1,
  "CompetitionDistance": 500,
  "StoreType": "a",
  "Assortment": "a",
  "SchoolHoliday": 0,
  "lag_1": 8.5,
  "lag_4": 8.3,
  "rolling_mean_4": 8.4
}
```

### Sample Request

```json
{
  "predicted_weekly_sales": 16205.53
}
```

## 📂 Project Structure

```
Retail-Price-Elasticity-and-Sales-Forecast/
│
├── app.py
├── requirements.txt
├── Model/
│ └── retail_weekly_forecast_model.pkl
├── dataset/
├── Model Training/
└── README.md
```


---

## ⚙️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- FastAPI  
- Uvicorn  
- Render (Cloud Deployment)

---

## 🧩 Key Learnings

- Time-series modeling requires chronological validation.
- Lag features significantly improve forecasting stability.
- Regularization (Ridge) helps manage multicollinearity from store dummies.
- The entire preprocessing pipeline should be serialized for production use.
- Version consistency of scikit-learn is critical when using Pickle.

---

## 📌 Future Improvements

- Add model monitoring & logging
- Add automated retraining pipeline
- Introduce feature importance analysis
- Deploy using Docker
- Add a frontend dashboard

---

## 👨‍💻 Author

**Shreyansh Pathak**  
Data Scientist | Retail Analytics | Machine Learning
