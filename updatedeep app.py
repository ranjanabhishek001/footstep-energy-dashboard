import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards

# --- Custom CSS ---
st.markdown('''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        :root {
            --primary: #2E86AB;
            --secondary: #F18F01;
            --accent: #C73E1D;
            --light: #F0F2F6;
            --dark: #2B2D42;
        }
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ed 100%);
        }
        
        h1, h2, h3 {
            color: var(--primary);
            font-weight: 600;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, var(--primary) 0%, #1a6f8b 100%);
            color: white;
            padding: 0.5em 2em;
            border-radius: 30px;
            border: none;
            font-weight: 500;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }
        
        .stSelectbox, .stNumberInput {
            border-radius: 10px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 20px;
            border-radius: 20px !important;
            background-color: white;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary) !important;
            color: white !important;
        }
        
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }
        
        .prediction-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.08);
            border-left: 5px solid var(--primary);
        }
    </style>
''', unsafe_allow_html=True)

# --- Title ---
st.title("👣 Footstep Energy Harvesting Dashboard")
st.markdown(""
    <div style="background: linear-gradient(135deg, #2E86AB 0%, #1a6f8b 100%); 
            padding: 15px; border-radius: 15px; color: white; margin-bottom: 30px;">
        <h3 style="color: white; margin: 0;">Predicting Energy Output from Footsteps using Machine Learning</h3>
    </div>
"", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("energy_harvesting_data.csv")

df = load_data()

# --- Initialize Models ---
@st.cache_resource
def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    return models

models = get_models()

# --- Feature Selection ---
X = df.drop(columns=["Energy_Output (mA)"])
y = df["Energy_Output (mA)"]

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Sidebar Info ---
st.sidebar.title("🔧 Settings")
model_option = st.sidebar.selectbox("Select Model", list(models.keys()))
show_all_models = st.sidebar.checkbox("Compare all models", value=True)

# --- Train Selected Model ---
model = models[model_option]
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# --- Evaluation Metrics ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred) if hasattr(model, 'predict') else None

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📈 Visualizations", "🔍 Model Comparison"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">'
                    f'<h3>R² Score</h3>'
                    f'<h2 style="color: var(--primary);">{r2:.3f}</h2>'
                    '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">'
                    f'<h3>RMSE</h3>'
                    f'<h2 style="color: var(--accent);">{rmse:.2f} mA</h2>'
                    '</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">'
                    f'<h3>MAE</h3>'
                    f'<h2 style="color: var(--secondary);">{mae:.2f} mA</h2>'
                    '</div>', unsafe_allow_html=True)
    
    style_metric_cards()
    
    st.markdown("---")
    
    # Predict from user input
    st.markdown("### 🔍 Make a Prediction")
    input_cols = st.columns(2)
    input_data = {}
    
    for i, col in enumerate(X.columns):
        with input_cols[i % 2]:
            input_data[col] = st.number_input(
                f"Enter {col}", 
                value=float(df[col].mean()),
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                step=0.1
            )
    
    if st.button("Predict Energy Output"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        if show_all_models:
            # Compare predictions from all models
            predictions = {}
            for name, m in models.items():
                m.fit(X_train_scaled, y_train)  # Retrain to ensure fairness
                predictions[name] = m.predict(input_scaled)[0]
            
            # Create comparison chart
            fig = go.Figure()
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            names = [x[0] for x in sorted_preds]
            values = [x[1] for x in sorted_preds]
            
            fig.add_trace(go.Bar(
                x=values,
                y=names,
                orientation='h',
                marker_color=['#2E86AB', '#F18F01', '#C73E1D'],
                text=[f"{v:.2f} mA" for v in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Model Comparison for Current Input",
                xaxis_title="Predicted Energy Output (mA)",
                yaxis_title="Model",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show best model
            best_model = max(predictions.items(), key=lambda x: x[1])
            st.markdown(f""
                <div class="prediction-card">
                    <h3>Best Model for This Input</h3>
                    <p style="font-size: 24px; margin: 10px 0;"><b>{best_model[0]}</b></p>
                    <p style="font-size: 18px;">Predicted Output: <b style="color: var(--accent);">{best_model[1]:.2f} mA</b></p>
                </div>
            "", unsafe_allow_html=True)
        else:
            # Single model prediction
            prediction = model.predict(input_scaled)[0]
            st.markdown(f""
                <div class="prediction-card">
                    <h3>Prediction Result</h3>
                    <p style="font-size: 24px; margin: 10px 0;"><b>{model_option}</b></p>
                    <p style="font-size: 18px;">Predicted Output: <b style="color: var(--accent);">{prediction:.2f} mA</b></p>
                </div>
            "", unsafe_allow_html=True)

with tab2:
    st.markdown("### 🔬 Data Exploration")
    
    # Data summary
    with st.expander("📋 Dataset Overview"):
        st.dataframe(df.describe().style.background_gradient(cmap='Blues'))
    
    # Correlation Heatmap
    st.markdown("#### 🔥 Correlation Heatmap")
    fig1 = px.imshow(
        df.corr(),
        text_auto=True,
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Feature Distribution
    st.markdown("#### 📊 Feature Distributions")
    feature = st.selectbox("Select feature to visualize", X.columns)
    
    fig_dist = px.histogram(
        df, 
        x=feature, 
        marginal="box",
        color_discrete_sequence=['#2E86AB'],
        title=f"Distribution of {feature}"
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Feature vs Energy Output
    st.markdown("#### ⚡ Feature vs Energy Output")
    fig_scatter = px.scatter(
        df,
        x=feature,
        y="Energy_Output (mA)",
        trendline="lowess",
        color_discrete_sequence=['#F18F01'],
        title=f"{feature} vs Energy Output"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Feature Importance (if tree-based model)
    if model_option in ["Random Forest", "XGBoost"]:
        st.markdown("#### 🧠 Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)
        
        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation='h',
            color="Importance",
            color_continuous_scale='Blues',
            title="Feature Importance"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

with tab3:
    st.markdown("### 🏆 Model Performance Comparison")
    
    # Train all models and collect metrics
    metrics = []
    for name, m in models.items():
        m.fit(X_train_scaled, y_train)
        y_pred = m.predict(X_test_scaled)
        metrics.append({
            "Model": name,
            "R²": r2_score(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred)
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Display metrics table
    st.dataframe(
        metrics_df.style
        .background_gradient(subset=["R²"], cmap='Greens')
        .background_gradient(subset=["RMSE", "MAE"], cmap='Reds_r')
        .format({"R²": "{:.3f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}),
        use_container_width=True
    )
    
    # Interactive comparison chart
    metric_to_compare = st.selectbox("Select metric to compare", ["R²", "RMSE", "MAE"])
    
    fig_compare = px.bar(
        metrics_df,
        x="Model",
        y=metric_to_compare,
        color="Model",
        color_discrete_sequence=['#2E86AB', '#F18F01', '#C73E1D'],
        text_auto='.2f',
        title=f"Model Comparison by {metric_to_compare}"
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # Actual vs Predicted comparison
    st.markdown("#### 📈 Actual vs Predicted Comparison")
    
    actual_vs_pred = []
    for name, m in models.items():
        m.fit(X_train_scaled, y_train)
        y_pred = m.predict(X_test_scaled)
        actual_vs_pred.append(pd.DataFrame({
            "Model": name,
            "Actual": y_test,
            "Predicted": y_pred
        }))
    
    actual_vs_pred_df = pd.concat(actual_vs_pred)
    
    fig_avp = px.scatter(
        actual_vs_pred_df,
        x="Actual",
        y="Predicted",
        color="Model",
        facet_col="Model",
        facet_col_wrap=3,
        color_discrete_sequence=['#2E86AB', '#F18F01', '#C73E1D'],
        trendline="lowess",
        title="Actual vs Predicted Values Across Models"
    )
    st.plotly_chart(fig_avp, use_container_width=True)


