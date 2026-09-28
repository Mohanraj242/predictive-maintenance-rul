# app/dashboard.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import shap
from sklearn.preprocessing import StandardScaler

# ---------------- Paths (bulletproof, relative to this file) ----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_lstm.pth")

SEQ_LENGTH = 30
FLAT = ['sensor_1','sensor_5','sensor_6','sensor_10','sensor_16','sensor_18','sensor_19']
COLS = ['unit_number','time_in_cycles','op_setting_1','op_setting_2','op_setting_3'] + \
       [f'sensor_{i}' for i in range(1, 22)]

# ---------------- Model (must match the architecture you trained) ----------------
class LSTMRUL(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# ---------------- Helper: sliding windows for ONE engine ----------------
def windows_for_engine(df_engine, sensor_cols):
    df_engine = df_engine.sort_values('time_in_cycles')
    data = df_engine[sensor_cols].values
    cycles = df_engine['time_in_cycles'].values
    X, cyc = [], []
    for i in range(len(data) - SEQ_LENGTH + 1):
        X.append(data[i:i + SEQ_LENGTH])
        cyc.append(cycles[i + SEQ_LENGTH - 1])
    return np.array(X), np.array(cyc)

# ---------------- Load everything ONCE and cache it ----------------
@st.cache_resource
def load_assets():
    train = pd.read_csv(os.path.join(RAW, 'train_FD001.txt'), sep='\s+', header=None, names=COLS)
    test  = pd.read_csv(os.path.join(RAW, 'test_FD001.txt'),  sep='\s+', header=None, names=COLS)

    train_c = train.drop(columns=FLAT)
    test_c  = test.drop(columns=FLAT)
    sensor_cols = [c for c in train_c.columns if c.startswith('sensor_')]

    # Fit scaler on TRAIN only (leakage prevention, same as notebook)
    scaler = StandardScaler()
    train_c[sensor_cols] = scaler.fit_transform(train_c[sensor_cols])
    test_c[sensor_cols]  = scaler.transform(test_c[sensor_cols])

    model = LSTMRUL(len(sensor_cols))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # SHAP background: sample windows from a few healthy train engines
    bg_list = []
    for uid in [1, 2, 3, 4, 5]:
        Xb, _ = windows_for_engine(train_c[train_c.unit_number == uid], sensor_cols)
        bg_list.append(Xb)
    background = np.concatenate(bg_list)
    rng = np.random.default_rng(42)
    background = background[rng.choice(len(background), 50, replace=False)]
    explainer = shap.GradientExplainer(model, torch.FloatTensor(background))

    return train, test, test_c, sensor_cols, model, explainer

# ================= DASHBOARD UI =================
st.set_page_config(page_title="Turbofan RUL Dashboard", layout="wide", page_icon="🛩️")
st.title("🛩️ Turbofan Predictive Maintenance Dashboard")
st.caption("Condition-based maintenance: predicting Remaining Useful Life (RUL) from live sensor trends.")

train, test, test_c, sensor_cols, model, explainer = load_assets()

unit = st.sidebar.selectbox("Select engine unit (in-service fleet)",
                            sorted(test['unit_number'].unique()))

eng_raw = test[test.unit_number == unit].sort_values('time_in_cycles')
eng_scaled = test_c[test_c.unit_number == unit].sort_values('time_in_cycles')

# ---------- 1) Sensor trends (RAW values, not scaled) ----------
st.subheader("1️ Key Sensor Trends")
default_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_11']
chosen = st.multiselect("Sensors to display", sensor_cols, default=default_sensors)

fig, ax = plt.subplots(figsize=(10, 4))
for s in chosen:
    ax.plot(eng_raw['time_in_cycles'], eng_raw[s], label=s)
ax.set_xlabel('Cycle')
ax.set_ylabel('Raw reading')
ax.set_title(f'Engine {unit} — Sensor Trends')
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# ---------- 2) Predicted RUL curve over time ----------
st.subheader("2️⃣ Predicted RUL Curve")
X, cyc = windows_for_engine(eng_scaled, sensor_cols)
with torch.no_grad():
    preds = np.clip(model(torch.FloatTensor(X)).numpy().flatten(), 0, 125)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(cyc, preds, color='blue', label='Predicted RUL')
ax.axhline(20, color='red', ls='--', label='Critical threshold (20 cycles)')
ax.set_xlabel('Cycle')
ax.set_ylabel('Predicted RUL (cycles)')
ax.set_title(f'Engine {unit} — RUL Trajectory')
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

latest_rul = float(preds[-1])

# ---------- 3) Red / Yellow / Green risk flag ----------
st.subheader("3️⃣ Maintenance Risk Flag")
if latest_rul <= 20:
    st.error(f"🔴 CRITICAL — Predicted RUL: {latest_rul:.0f} cycles. Schedule maintenance IMMEDIATELY.")
elif latest_rul <= 60:
    st.warning(f"🟡 MONITOR — Predicted RUL: {latest_rul:.0f} cycles. Plan a maintenance window.")
else:
    st.success(f"🟢 HEALTHY — Predicted RUL: {latest_rul:.0f} cycles. No action needed.")

# ---------- 4) SHAP explanation for the latest prediction ----------
st.subheader("4️⃣ Why? — SHAP Explanation (latest window)")
sv = np.array(explainer.shap_values(torch.FloatTensor(X[-1:]))).squeeze()
imp = np.abs(sv).mean(axis=0)
order = np.argsort(imp)[::-1]

fig, ax = plt.subplots(figsize=(10, 4))
colors = ['crimson'] * 3 + ['teal'] * (len(order) - 3)
ax.barh([sensor_cols[i] for i in order], imp[order], color=colors)
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP value| (impact on prediction)')
ax.set_title(f'Engine {unit} — Top drivers of the latest prediction')
ax.grid(alpha=0.3, axis='x')
st.pyplot(fig)