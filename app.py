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

st.set_page_config(page_title="ML Dashboard", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #F0F2F6;
            color: #262730;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
        }
        .stSelectbox, .stTextInput, .stNumberInput {
            background-color: #ffffff;
            border: 1px solid #ccc;
            border-radius: 10px;
        }
        h1, h2, h3, h4 {
            color: #003366;
        }
        .stDataFrame {
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Displacement Force Prediction Dashboard")
st.write("Upload your dataset with features like footstep frequency, weight, and target displacement force.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset:")
    st.dataframe(df)

    if df.isnull().sum().sum() > 0:
        st.warning("Dataset contains missing values. They will be dropped.")
        df = df.dropna()

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox("Select the target column", numeric_columns)
    features = st.multiselect("Select feature columns", [col for col in numeric_columns if col != target])

    if features:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("Model Training & Evaluation")

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "XGBoost": XGBRegressor()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.markdown(f"**{name}** - R²: {r2:.4f}, MSE: {mse:.4f}")

        st.subheader("Try It Yourself!")
        user_input = [st.number_input(f"Enter {col}", value=float(X[col].mean())) for col in features]
        chosen_model_name = st.selectbox("Choose a model for prediction", list(models.keys()))
        chosen_model = models[chosen_model_name]
        prediction = chosen_model.predict([user_input])[0]
        st.success(f"Predicted Displacement Force: {prediction:.2f}")

        st.header("Visualizations and Insights")

        if st.checkbox("Missing Values Heatmap"):
            fig, ax = plt.subplots()
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
            st.pyplot(fig)
            st.info("Highlights where data is missing. Helps identify gaps that may need preprocessing.")

        if st.checkbox("Correlation Heatmap"):
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(fig)
            st.info("Correlation heatmap shows relationships between numeric features. High absolute values (close to 1 or -1) indicate strong correlations.")

        if st.checkbox("Pairplot"):
            fig = sns.pairplot(df[numeric_columns])
            st.pyplot(fig)
            st.info("Pairplot helps visually assess relationships between all numeric variables, aiding in understanding patterns and potential outliers.")

        if st.checkbox("Box Plot for Each Feature"):
            for col in numeric_columns:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)
                st.info(f"Box plot for **{col}** shows distribution and outliers.")

        if st.checkbox("Distribution Plots"):
            for col in numeric_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                st.pyplot(fig)
                st.info(f"Distribution of **{col}** shows its frequency and skewness.")

        if st.checkbox("Feature Importance (Random Forest)"):
            model_rf = RandomForestRegressor()
            model_rf.fit(X, y)
            importances = pd.Series(model_rf.feature_importances_, index=X.columns)
            fig, ax = plt.subplots()
            importances.sort_values().plot(kind='barh', ax=ax)
            st.pyplot(fig)
            st.info("Feature importance shows which inputs influence the prediction most in the Random Forest model.")

        if st.checkbox("Dendrogram - Hierarchical Clustering"):
            import scipy.cluster.hierarchy as sch
            from scipy.spatial.distance import pdist

            numeric_data = df.select_dtypes(include=np.number).dropna()
            distance_matrix = pdist(numeric_data)
            linkage_matrix = sch.linkage(distance_matrix, method='ward')

            fig, ax = plt.subplots(figsize=(12, 6))
            sch.dendrogram(linkage_matrix, ax=ax)
            st.pyplot(fig)
            st.info("The dendrogram shows how samples are hierarchically clustered based on similarity. Useful for understanding nested clusters and optimal group counts.")

        if st.checkbox("SHAP Values - Feature Importance (XGBoost)"):
            import shap
            shap.initjs()

            df_clean = df.dropna()
            X = df_clean.select_dtypes(include=np.number).drop(columns=[df_clean.columns[-1]])
            y = df_clean[df_clean.columns[-1]]

            model = XGBRegressor()
            model.fit(X, y)
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

            st.subheader("SHAP Summary Plot")
            fig_summary = shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(bbox_inches='tight', dpi=300)
            st.info("SHAP values explain the contribution of each feature to a prediction. The beeswarm plot highlights which features have the biggest impact across the dataset.")

        if st.checkbox("UMAP Projection"):
            import umap
            numeric_data = df.select_dtypes(include=np.number).dropna()
            reducer = umap.UMAP(random_state=42)
            embedding = reducer.fit_transform(numeric_data)
            umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
            fig = px.scatter(umap_df, x='UMAP1', y='UMAP2', title="UMAP Dimensionality Reduction")
            st.plotly_chart(fig)
            st.info("UMAP projects high-dimensional data into 2D while preserving the global structure and clustering. Great for visualizing complex datasets.")

        if st.checkbox("RadViz Visualization"):
            from pandas.plotting import radviz
            df_clean = df.dropna()
            if df_clean.select_dtypes(include='object').shape[1] > 0:
                target_col = df_clean.select_dtypes(include='object').columns[0]
                fig, ax = plt.subplots()
                radviz(df_clean, target=target_col, ax=ax)
                st.pyplot(fig)
                st.info("RadViz helps visualize multivariate data by mapping numeric features to points on a circle. Good for spotting clusters across categorical targets.")
            else:
                st.warning("RadViz requires at least one categorical column in your dataset.")

        if st.checkbox("Interactive Feature Filter"):
            numeric_cols = df.select_dtypes(include=np.number).columns
            selected_col = st.selectbox("Select feature to filter", numeric_cols)
            min_val = float(df[selected_col].min())
            max_val = float(df[selected_col].max())
            user_range = st.slider("Filter range", min_val, max_val, (min_val, max_val))
            filtered_df = df[(df[selected_col] >= user_range[0]) & (df[selected_col] <= user_range[1])]
            st.dataframe(filtered_df)
            st.info(f"Filtered data based on {selected_col} values between {user_range[0]} and {user_range[1]}. Useful for interactive exploration.")
