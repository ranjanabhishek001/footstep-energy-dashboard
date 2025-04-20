import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Energy from Footsteps", layout="wide")

# Load data
data = pd.read_csv("energy_harvesting_data_synthetic.csv")
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

# Train models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
}

for name, model in models.items():
    model.fit(X_train, y_train)

# Main interface
st.title("⚡ Energy Prediction from Footsteps")
model_choice = st.selectbox("Choose a Machine Learning Model", list(models.keys()))
model = models[model_choice]

if st.button("Predict Energy Output"):
    prediction = model.predict(input_data)[0]
    st.success(f"🔋 Predicted Energy Output: **{prediction:.2f} mA**")

# ---------------------- Visualizations ----------------------
st.subheader("📊 Visual Insights")

tab1, tab2 = st.tabs(["📌 Overview", "📊 Model Evaluation"])

with tab1:
    st.markdown("### 🔥 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("### 🎯 Histogram of Energy Output")
    fig2 = px.histogram(data, x="Energy_Output (mA)", nbins=30, color_discrete_sequence=["#2E86AB"])
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🧊 Boxplot: User Weight vs Energy Output")
    fig3 = px.box(data, x="User_Weight (kg)", y="Energy_Output (mA)", points="all", color_discrete_sequence=["#FF5733"])
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    if model_choice in ["Random Forest", "XGBoost"]:
        st.markdown("### 💡 Feature Importances")
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        fig4 = px.bar(importance_df, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale='Blues')
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 🎯 Predicted vs Actual")
    y_pred = model.predict(X_test)
    fig5 = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted Energy Output")
    fig5.add_shape(type="line", x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(),
                   line=dict(dash='dash', color="red"))
    st.plotly_chart(fig5, use_container_width=True)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    st.metric(label="R² Score", value=f"{r2:.3f}")
    st.metric(label="MSE", value=f"{mse:.2f}")
