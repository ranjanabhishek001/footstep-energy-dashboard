import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Energy from Footsteps", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        body { background-color: #f4f6f9; color: #333; }
        .main { background-color: #ffffff; padding: 2rem; border-radius: 12px; box-shadow: 0px 0px 15px rgba(0,0,0,0.05); }
        h1, h2, h3 { color: #2E86AB; }
        .stButton button { border-radius: 8px; background-color: #2E86AB; color: white; }
    </style>
""", unsafe_allow_html=True)

# Load data
data = pd.read_csv("energy_harvesting_data.csv")

X = data.drop("Energy_Output (mA)", axis=1)
y = data["Energy_Output (mA)"]

# Sidebar inputs
st.sidebar.header("Input Parameters")
step_frequency = st.sidebar.slider("Step Frequency (steps/sec)", 0.5, 5.0, 2.0)
foot_pressure = st.sidebar.slider("Foot Pressure (N)", 50, 500, 200)
stride_length = st.sidebar.slider("Stride Length (m)", 0.3, 1.5, 0.8)
user_weight = st.sidebar.slider("User Weight (kg)", 30, 150, 70)
displacement_force = st.sidebar.slider("Displacement Force (N)", 10, 200, 100)

input_data = pd.DataFrame({
    'Step_Frequency (steps/sec)': [step_frequency],
    'Foot_Pressure (N)': [foot_pressure],
    'Stride_Length (m)': [stride_length],
    'User_Weight (kg)': [user_weight],
    'Displacement_Force (N)': [displacement_force]
})

# Train models (or load pre-trained models)
def train_and_save_models():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    linear = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

    linear.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    joblib.dump(linear, "linear_model.pkl")
    joblib.dump(rf, "rf_model.pkl")
    joblib.dump(xgb, "xgb_model.pkl")

if not all(os.path.exists(f"models/{m}_model.pkl") for m in ["linear", "rf", "xgb"]):
    train_and_save_models()

# Load models
model_dict = {
    "Linear Regression": joblib.load("models/linear_model.pkl"),
    "Random Forest": joblib.load("models/rf_model.pkl"),
    "XGBoost": joblib.load("models/xgb_model.pkl"),
}

# Main interface
st.title("⚡ Energy Prediction from Footsteps")

model_choice = st.selectbox("Choose a Machine Learning Model", list(model_dict.keys()))
model = model_dict[model_choice]

if st.button("Predict Energy Output"):
    prediction = model.predict(input_data)[0]
    st.success(f"🔋 Predicted Energy Output: **{prediction:.2f} mA**")

# Visualizations
st.subheader("📊 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with col2:
    st.markdown("### Scatter Plot: Displacement Force vs Energy Output")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=data["Displacement_Force (N)"], y=data["Energy_Output (mA)"], hue=data["User_Weight (kg)"], palette="viridis", ax=ax2)
    st.pyplot(fig2)

# Feature importance
if model_choice in ["Random Forest", "XGBoost"]:
    st.markdown("### 🔍 Feature Importances")
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    fig3, ax3 = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax3)
    st.pyplot(fig3)
