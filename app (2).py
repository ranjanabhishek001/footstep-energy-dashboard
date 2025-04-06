import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- Custom CSS ---
st.markdown('''
    <style>
        .main {
            background-color: #f0f2f6;
        }
        h1, h2 {
            color: #2E86AB;
        }
        .stButton>button {
            background-color: #2E86AB;
            color: white;
            padding: 0.5em 1em;
            border-radius: 10px;
            border: none;
        }
    </style>
''', unsafe_allow_html=True)

# --- Title ---
st.title("👣 Footstep Energy Harvesting Dashboard")
st.subheader("Predicting Energy Output using ML Models")

# --- Load Data ---
df = pd.read_csv("energy_harvesting_data.csv")

# --- Sidebar Info ---
st.sidebar.title("🔧 Settings")
model_option = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest", "XGBoost"])

# --- Feature Selection ---
X = df.drop(columns=["Energy_Output (mA)"])
y = df["Energy_Output (mA)"]

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---
if model_option == "Linear Regression":
    model = LinearRegression()
elif model_option == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# --- Evaluation Metrics ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Predictions", "📈 Visualizations"])

with tab1:
    st.markdown("### 📋 Model Evaluation")
    st.write(f"**Model Used**: {model_option}")
    st.write(f"**R² Score**: {r2:.4f}")
    st.write(f"**MSE**: {mse:.4f}")
    
    # Predict from user input
    st.markdown("### 🔍 Make a Prediction")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))
    
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.success(f"🔋 Predicted Energy Output: **{prediction:.2f} mA**")

with tab2:
    st.markdown("### 🔬 Data Distribution")
    st.dataframe(df.head())

    # Correlation Heatmap
    st.markdown("#### 🔥 Correlation Heatmap")
    fig1, ax1 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    # Feature Importance (if tree-based model)
    if model_option in ["Random Forest", "XGBoost"]:
        st.markdown("#### 🧠 Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        fig2, ax2 = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax2)
        st.pyplot(fig2)


