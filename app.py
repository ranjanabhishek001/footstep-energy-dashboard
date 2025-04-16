import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from xgboost import XGBRegressor, XGBClassifier
import plotly.express as px
import serial
import time

st.set_page_config(page_title="Smart AutoML Dashboard", layout="wide")
st.title("🤖 Smart AutoML Dashboard")

# Optional live sensor data section
st.sidebar.subheader("📡 Live Sensor Data")
live_mode = st.sidebar.checkbox("Enable Live Sensor Feed")

if live_mode:
    try:
        port = st.sidebar.text_input("Enter COM port (e.g., COM3 or /dev/ttyUSB0)", value="COM3")
        baud = st.sidebar.number_input("Baud Rate", value=9600)
        duration = st.sidebar.slider("How many seconds to collect?", 1, 30, 5)

        if st.sidebar.button("Start Reading Sensor"):
            ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            st.write("Reading from Arduino...")
            data = []
            start_time = time.time()
            with st.spinner("Collecting data..."):
                while time.time() - start_time < duration:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8').strip()
                        st.text(f"Raw: {line}")
                        try:
                            voltage = float(line)
                            timestamp = time.time()
                            data.append([timestamp, voltage])
                        except:
                            pass
            ser.close()
            if data:
                df_live = pd.DataFrame(data, columns=['Timestamp', 'Voltage'])
                st.subheader("📊 Live Voltage Data")
                st.line_chart(df_live.set_index('Timestamp'))
                st.session_state.live_data = df_live
    except Exception as e:
        st.error(f"Error: {e}")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Visualize Dataset")
    if st.checkbox("Show Missing Values Heatmap"):
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        st.pyplot(fig)

    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.subheader("Select Target Column")
    target_col = st.selectbox("Target (what you want to predict)", df.columns)

    if target_col:
        all_features = df.drop(columns=[target_col]).columns.tolist()
        selected_features = st.multiselect("Select Features to Include in Model", all_features, default=all_features)

        X = df[selected_features]
        y = df[target_col]

        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        task_type = "Regression" if len(np.unique(y)) > 10 and y.dtype != "object" else "Classification"
        st.markdown(f"### 🔍 Detected Problem Type: **{task_type}**")

        if task_type == "Regression":
            model_option = st.selectbox("Choose a regression model", [
                "Linear Regression", "Random Forest Regressor", "XGBoost Regressor"])
            if model_option == "Linear Regression":
                model = LinearRegression()
            elif model_option == "Random Forest Regressor":
                n_estimators = st.slider("n_estimators (Random Forest)", 10, 200, 100)
                model = RandomForestRegressor(n_estimators=n_estimators)
            else:
                n_estimators = st.slider("n_estimators (XGBoost)", 10, 200, 100)
                model = XGBRegressor(objective="reg:squarederror", n_estimators=n_estimators)
        else:
            model_option = st.selectbox("Choose a classification model", [
                "Logistic Regression", "Random Forest Classifier", "XGBoost Classifier"])
            if model_option == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_option == "Random Forest Classifier":
                n_estimators = st.slider("n_estimators (Random Forest)", 10, 200, 100)
                model = RandomForestClassifier(n_estimators=n_estimators)
            else:
                n_estimators = st.slider("n_estimators (XGBoost)", 10, 200, 100)
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=n_estimators)

        if st.button("Train Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("📈 Model Performance")

            if task_type == "Regression":
                st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
                st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
                fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
                st.plotly_chart(fig)
            else:
                st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)

                if len(np.unique(y_test)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("ROC Curve")
                    ax.legend()
                    st.pyplot(fig)

            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
                st.plotly_chart(fig)
