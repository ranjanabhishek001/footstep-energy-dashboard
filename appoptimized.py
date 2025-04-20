import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(page_title="Footstep Energy Prediction Dashboard", layout="wide")

st.title("⚡ Footstep Energy Prediction using Machine Learning")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your dataset (CSV only)", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Preview of Uploaded Dataset")
    st.dataframe(df.head())

    # --- Dynamic Output Column Detection ---
    try:
        output_col = [col for col in df.columns if "Energy" in col][0]
        st.success(f"✅ Detected target column: `{output_col}`")
    except IndexError:
        st.error("❌ Could not find a column containing the word 'Energy'. Please check your dataset.")
        st.stop()

    # Feature and target split
    X = df.drop(output_col, axis=1)
    y = df[output_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Training ---
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            "Model": model,
            "R2 Score": r2,
            "MAE": mae,
            "RMSE": rmse,
            "y_pred": y_pred
        }

    # --- Performance Metrics Table ---
    st.write("### 📈 Model Performance Comparison")

    perf_df = pd.DataFrame({
        model: {
            "R² Score": res["R2 Score"],
            "MAE": res["MAE"],
            "RMSE": res["RMSE"]
        }
        for model, res in results.items()
    }).T

    st.dataframe(perf_df.style.background_gradient(cmap='Blues', axis=0))

    # --- Visualization ---
    st.write("### 📉 Actual vs Predicted Plot")

    selected_model = st.selectbox("Select a model to visualize", list(results.keys()))
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label='Actual', marker='o')
    ax.plot(results[selected_model]['y_pred'], label='Predicted', marker='x')
    ax.set_title(f'Actual vs Predicted - {selected_model}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel(output_col)
    ax.legend()
    st.pyplot(fig)

    # --- Optional Prediction Input ---
    st.write("### 🔍 Make a New Prediction")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))

    input_df = pd.DataFrame([input_data])
    pred_value = results[selected_model]["Model"].predict(input_df)[0]
    st.success(f"Predicted {output_col}: **{pred_value:.2f}**")
