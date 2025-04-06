import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page settings
st.set_page_config(page_title="Energy Prediction Dashboard", layout="wide")
st.title("🔋 Footstep Energy Harvesting Dashboard")
st.markdown(""
<style>
body {
    background-color: #f5f7fa;
}
.big-font {
    font-size:20px;
}
</style>
"", unsafe_allow_html=True)

# Upload data
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head())

    # Feature Selection
    X = df.drop(columns=['Energy_Output'])
    y = df['Energy_Output']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {'model': model, 'MSE': mse, 'R2': r2}

    # Results Table
    st.subheader("📊 Model Performance")
    res_df = pd.DataFrame({k: {'MSE': v['MSE'], 'R2 Score': v['R2']} for k, v in results.items()}).T
    st.dataframe(res_df)

    # Visualizations
    st.subheader("📈 Actual vs Predicted")
    selected_model = st.selectbox("Choose a model to visualize", list(results.keys()))
    selected = results[selected_model]
    y_pred = selected['model'].predict(X_test)

    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Energy Output (mA)")
    ax.set_ylabel("Predicted Energy Output (mA)")
    ax.set_title(f"Actual vs Predicted - {selected_model}")
    st.pyplot(fig)

    # Feature Importance for RF/XGB
    if selected_model != 'Linear Regression':
        st.subheader("🧠 Feature Importance")
        importance = selected['model'].feature_importances_
        imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
        fig2, ax2 = plt.subplots()
        sns.barplot(data=imp_df, x='Importance', y='Feature', ax=ax2)
        st.pyplot(fig2)

    # Prediction Interface
    st.subheader("🔮 Predict Energy Output")
    with st.form("prediction_form"):
        user_input = {}
        for feature in X.columns:
            user_input[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
        submit = st.form_submit_button("Predict")

    if submit:
        user_df = pd.DataFrame([user_input])
        prediction = selected['model'].predict(user_df)[0]
        st.success(f"Predicted Energy Output: {prediction:.2f} mA")

