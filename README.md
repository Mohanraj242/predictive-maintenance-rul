# 🛩️ Predictive Maintenance & RUL Estimation

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predictive-maintenance-rul-5bcc6txseqfgmgq7itsyj8.streamlit.app/)

🚀 **Live Demo:** https://predictive-maintenance-rul-5bcc6txseqfgmgq7itsyj8.streamlit.app/
End-to-end ML project predicting Remaining Useful Life (RUL) of turbofan engines
using the NASA C-MAPSS dataset, deployed as a Streamlit dashboard.

## 🎯 Business Goal
Move from calendar-based or failure-based maintenance to **condition-based maintenance**.

## 📊 Results
| Model | Test RMSE (cycles) |
|-------|--------------------|
| XGBoost (baseline) | 19.26 |
| **LSTM (PyTorch)** | **15.89** |

## 🧰 Tech Stack
Python, PyTorch, XGBoost, scikit-learn, SHAP, Streamlit, pandas, matplotlib

## 🚀 How to Run
1. `pip install -r requirements.txt`
2. `streamlit run app/dashboard.py`

## 🔑 Methodology Highlights
- Split by engine unit + scaler fit on train only (no data leakage)
- RUL capped at 125 cycles (standard C-MAPSS practice)
- SHAP GradientExplainer for per-sensor explainability