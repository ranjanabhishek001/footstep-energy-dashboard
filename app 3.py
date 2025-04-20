import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Energy Output Prediction", layout="wide")
st.title("⚡ Energy Prediction from Footsteps using Machine Learning")

st.sidebar.header("📂 Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.subheader("📄 Uploaded Dataset Preview")
    st.write(df.head())

    # Data Splitting
    X = df.drop("Energy_Output (mA)", axis=1)
    y = df["Energy_Output (mA)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42)
    }

    st.sidebar.header("⚙️ Model Selection")
    model_option = st.sidebar.selectbox("Select a model", list(models.keys()))
    model = models[model_option]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Tabs for Results and Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Results", "📊 Visualizations", "📌 All Model Comparison", "🧪 Live Prediction"])

    with tab1:
        st.subheader("📊 Model Performance")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**Mean Squared Error:** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        st.subheader("📋 Prediction Table")
        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).reset_index(drop=True)
        st.dataframe(pred_df)

        csv_pred = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction Table", data=csv_pred, file_name="prediction_results.csv", mime="text/csv")

    with tab2:
        st.subheader("📉 Correlation Heatmap")
        fig1, ax1 = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax1)
        st.pyplot(fig1)

        st.subheader("📌 Actual vs Predicted Plot")
        actual_vs_pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).reset_index(drop=True)
        fig_actual_pred = px.scatter(
            actual_vs_pred_df, x="Actual", y="Predicted",
            title=f"🎯 Actual vs Predicted Energy Output - {model_option}",
            labels={"Actual": "Actual Energy Output (mA)", "Predicted": "Predicted Energy Output (mA)"},
            trendline="ols", color_discrete_sequence=["#00cc96"]
        )
        fig_actual_pred.update_layout(showlegend=False)
        st.plotly_chart(fig_actual_pred, use_container_width=True)

    with tab3:
        st.subheader("📌 Actual vs Predicted Energy Output (All Models)")
        all_preds = pd.DataFrame({"Actual": y_test.reset_index(drop=True)})
        model_metrics = []

        for name, mdl in models.items():
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)
            all_preds[name] = preds
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            model_metrics.append({"Model": name, "MSE": mse, "R² Score": r2})

        fig_all_models = px.line(all_preds, labels={"value": "Energy Output (mA)", "index": "Sample Index"})
        fig_all_models.update_layout(title="📊 Actual vs Predicted Energy Output (All Models)",
                                     legend_title_text='Legend')
        st.plotly_chart(fig_all_models, use_container_width=True)

        st.subheader("📌 Side-by-Side Model Performance")
        metrics_df = pd.DataFrame(model_metrics)
        st.dataframe(metrics_df.style.highlight_max(axis=0))

        csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Model Metrics", data=csv_metrics, file_name="model_comparison.csv", mime="text/csv")

    with tab4:
        st.subheader("🧮 Predict Energy Output from Manual Input")

        feature_inputs = {}
        for col in X.columns:
            val = st.number_input(f"Enter {col}", value=float(df[col].mean()), format="%.2f")
            feature_inputs[col] = val

        if st.button("🚀 Predict Energy Output"):
            input_df = pd.DataFrame([feature_inputs])
            predicted_energy = model.predict(input_df)[0]
            st.success(f"⚡ Predicted Energy Output: **{predicted_energy:.2f} mA**")
else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
