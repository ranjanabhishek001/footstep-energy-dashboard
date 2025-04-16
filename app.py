import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Energy Generation Dashboard", layout="wide")
st.title("Energy Generation Prediction Dashboard")
st.write("Upload your dataset containing features like footstep frequency, weight, and target energy output.")

# Function to load the data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    
    # Handle categorical data (Label Encoding)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    encoder = LabelEncoder()

    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])  # Label Encoding for categorical columns

    return df.dropna()

# Function to train models
@st.cache_resource
def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Preview of Dataset:")
    st.dataframe(df)

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox("Select the target column", numeric_columns)
    features = st.multiselect("Select feature columns", [col for col in numeric_columns if col != target])

    if features:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("Model Training & Evaluation")

        models = train_models(X_train, y_train)

        for name, model in models.items():
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.markdown(f"**{name}** - R²: {r2:.4f}, MSE: {mse:.4f}")

        st.subheader("Try It Yourself!")
        user_input = [st.number_input(f"Enter {col}", value=float(X[col].mean())) for col in features]
        chosen_model_name = st.selectbox("Choose a model for prediction", list(models.keys()))
        chosen_model = models[chosen_model_name]
        prediction = chosen_model.predict([user_input])[0]
        st.success(f"Predicted Energy Output: {prediction:.2f}")

        st.header("Visualizations and Insights")

        if st.checkbox("Missing Values Heatmap"):
            fig, ax = plt.subplots()
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
            st.pyplot(fig)
            st.info("This heatmap shows where data is missing, which can help identify gaps in the dataset.")

        if st.checkbox("Correlation Heatmap"):
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(fig)
            st.info("The correlation heatmap shows the relationships between numeric features.")

        if st.checkbox("Pairplot"):
            fig = sns.pairplot(df[numeric_columns])
            st.pyplot(fig)
            st.info("A pairplot displays relationships between numeric variables and helps detect potential outliers.")

        if st.checkbox("Box Plot for Each Feature"):
            for col in numeric_columns:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)
                st.info(f"Box plot for **{col}** shows the distribution and highlights any outliers.")

        if st.checkbox("Distribution Plots"):
            for col in numeric_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                st.pyplot(fig)
                st.info(f"Distribution of **{col}** shows its frequency and skewness, which is important for understanding data.")

        if st.checkbox("Feature Importance (Random Forest)"):
            model_rf = RandomForestRegressor()
            model_rf.fit(X, y)
            importances = pd.Series(model_rf.feature_importances_, index=X.columns)
            fig, ax = plt.subplots()
            importances.sort_values().plot(kind='barh', ax=ax)
            st.pyplot(fig)
            st.info("Feature importance from the Random Forest model indicates which features most influence the target variable.")

        if st.checkbox("Interactive Scatter Plot"):
            fig = px.scatter(df, x=features[0], y=target, title=f"Scatter plot between {features[0]} and {target}")
            st.plotly_chart(fig)
            st.info(f"Interactive scatter plot between **{features[0]}** and **{target}** helps visualize the relationship.")

        if st.checkbox("Energy Output vs Footstep Frequency"):
            fig = px.scatter(df, x='Step_Frequency (steps/sec)', y='Energy_Output (mA)', title="Energy Output vs Footstep Frequency")
            st.plotly_chart(fig)
            st.info("This plot helps visualize how footstep frequency correlates with energy output.")

