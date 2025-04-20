import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error

# Set Streamlit page config
st.set_page_config(page_title="Energy Output Predictor", layout="wide")

st.title("⚡ Energy Prediction from Footsteps using ML")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 ML Dashboard", "📁 Upload Dataset", "🧪 Model Diagnostics"])

# Load or Upload Dataset
@st.cache_data
def load_data():
    return pd.read_csv("energy_harvesting_data.csv")

df = load_data()

with tab2:
    st.header("📁 Upload Your Custom Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Custom dataset uploaded successfully!")

    st.write("### Preview of Dataset")
    st.dataframe(df.head())

# Feature selection
X = df.drop("Energy_Output", axis=1)
y = df["Energy_Output"]

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selection
model_name = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Random Forest", "XGBoost"])

if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif model_name == "XGBoost":
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# === TAB 1: ML Dashboard ===
with tab1:
    st.header("📊 ML Dashboard")
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("📈 Model Performance on Test Data")
    st.metric("R² Score", f"{r2:.3f}")
    st.metric("RMSE", f"{rmse:.2f} mA")

    # Actual vs Predicted Plot
    fig1 = px.scatter(x=y_test, y=y_pred,
                      labels={"x": "Actual Energy Output (mA)", "y": "Predicted Energy Output (mA)"},
                      title="Actual vs Predicted Energy Output")
    fig1.add_shape(type='line',
                   x0=y_test.min(), y0=y_test.min(),
                   x1=y_test.max(), y1=y_test.max(),
                   line=dict(color="red", dash="dash"))
    st.plotly_chart(fig1, use_container_width=True)

# === TAB 3: Model Diagnostics ===
with tab3:
    st.header("🧪 Model Diagnostics")

    # Training Evaluation
    train_pred = model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

    st.subheader("🔍 Training Performance")
    st.write(f"**Training R² Score:** {train_r2:.3f}")
    st.write(f"**Training RMSE:** {train_rmse:.2f} mA")

    # Cross Validation
    st.subheader("🔁 Cross-Validation (5-Fold)")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    st.write(f"**Cross-validated R² Scores:** {cv_scores}")
    st.write(f"**Mean CV R²:** {np.mean(cv_scores):.3f}")

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        st.subheader("📌 Feature Importances")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(importance_df)

        fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
        st.plotly_chart(fig_importance, use_container_width=True)

    # Actual vs Predicted
    st.subheader("🎯 Actual vs Predicted (Test Set)")
    fig2 = px.scatter(x=y_test, y=y_pred,
                      labels={'x': 'Actual Energy Output (mA)', 'y': 'Predicted Energy Output (mA)'},
                      title='Actual vs Predicted Energy Output')
    fig2.add_shape(type='line',
                   x0=y_test.min(), y0=y_test.min(),
                   x1=y_test.max(), y1=y_test.max(),
                   line=dict(color='green', dash='dot'))
    st.plotly_chart(fig2, use_container_width=True)

    # Feature Distributions
    st.subheader("📈 Feature Distributions")
    for col in X.columns:
        fig = px.histogram(df, x=col, nbins=30, title=f"{col} Distribution")
        st.plotly_chart(fig, use_container_width=True)
