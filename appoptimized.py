
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

        .info-icon {
            cursor: help;
            font-size: 0.8em;
            color: var(--primary);
        }
    </style>
''', unsafe_allow_html=True)

# --- Title ---
st.title('👣 Footstep Energy Harvesting Dashboard')
st.markdown('''
    <div style='background: linear-gradient(135deg, #2E86AB 0%, #1a6f8b 100%);
            padding: 15px; border-radius: 15px; color: white; margin-bottom: 30px;'>
        <h3 style='color: white; margin: 0;'>Predicting Energy Output from Footsteps using Machine Learning</h3>
    </div>
''', unsafe_allow_html=True)

# --- Cached Data Operations ---
@st.cache_data
def load_data():
    return pd.read_csv('energy_harvesting_data.csv')

@st.cache_data
def preprocess_data(df):
    X = df.drop(columns=['Energy_Output (mA)'])
    y = df['Energy_Output (mA)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    return X_train, X_test, y_train, y_test, scaler

# --- Model Training with Cache ---
@st.cache_resource
def train_models():
    X_train, X_test, y_train, y_test, scaler = preprocess_data(load_data())
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models, X_test, y_test, scaler

models, X_test, y_test, scaler = train_models()

# --- Sidebar ---
st.sidebar.title('🔧 Settings')
model_option = st.sidebar.selectbox('Select Model', list(models.keys()))
show_all_models = st.sidebar.checkbox('Compare all models', value=True)

# --- Theory Explanations ---
METRIC_EXPLANATIONS = {
    'R² Score': "Explains variance in data (0-1, higher=better)",
    'RMSE': "Average prediction error in mA (lower=better)",
    'MAE': "Absolute average error (robust to outliers)"
}

MODEL_DESCRIPTIONS = {
    'Linear Regression': "Best for linear relationships, fast but less flexible",
    'Random Forest': "Handles non-linear data, robust to outliers",
    'XGBoost': "Advanced boosting algorithm, best for complex patterns"
}

# --- Prediction Tab ---
tab1, tab2, tab3 = st.tabs(['📊 Predictions', '📈 Visualizations', '🔍 Model Comparison'])

with tab1:
    # Metrics Cards with Tooltips
    col1, col2, col3 = st.columns(3)
    with col1:
        y_pred = models[model_option].predict(X_test)
        st.markdown(f'''<div class='metric-card'>
            <h3>R² Score <span class='info-icon' title="{METRIC_EXPLANATIONS['R² Score']}">ℹ️</span></h3>
            <h2 style='color: var(--primary);'>{r2_score(y_test, y_pred):.3f}</h2>
        </div>''', unsafe_allow_html=True)
    
    # Similar columns for RMSE and MAE...

    # Prediction Inputs
    st.markdown('### 🔍 Make a Prediction')
    input_cols = st.columns(2)
    input_data = {}
    for i, col in enumerate(models[model_option].feature_names_in_):
        with input_cols[i % 2]:
            input_data[col] = st.number_input(
                f'Enter {col}', 
                value=float(load_data()[col].mean()),
                step=0.1
            )

    if st.button('Predict Energy Output'):
        input_scaled = scaler.transform(pd.DataFrame([input_data]))
        
        if show_all_models:
            predictions = {name: model.predict(input_scaled)[0] for name, model in models.items()}
            # Visualization code...

# --- Optimized Visualization Tab ---
with tab2:
    st.markdown('### 🔬 Data Exploration')
    with st.expander("📘 Understanding Visualizations"):
        st.markdown('''
        - **Correlation Heatmap**: Shows relationships between features (-1 to 1)
        - **Feature Importance**: Relative impact of each input on predictions
        - **Actual vs Predicted**: Ideal points fall on the diagonal line
        ''')
    
    # Cached visualizations
    @st.cache_data
    def create_correlation_plot(df):
        return px.imshow(df.corr(), color_continuous_scale='RdBu')

    st.plotly_chart(create_correlation_plot(load_data()), use_container_width=True)

# --- Model Comparison Tab ---
with tab3:
    st.markdown('### 🏆 Model Performance Comparison')
    with st.expander("ℹ️ Model Selection Guide"):
        st.markdown('''
        - **Linear Regression**: Use when relationships are simple/linear
        - **Tree-based Models**: Better for complex, non-linear patterns
        - Compare RMSE/MAE for error magnitude, R² for variance explanation
        ''')
    
    # Pre-computed metrics
    @st.cache_data
    def get_model_metrics(models):
        return [
            {
                'Model': name,
                'R²': r2_score(y_test, model.predict(X_test)),
                'RMSE': np.sqrt(mean_squared_error(y_test, model.predict(X_test))),
                'MAE': mean_absolute_error(y_test, model.predict(X_test))
            } 
            for name, model in models.items()
        ]
    
    metrics_df = pd.DataFrame(get_model_metrics(models))
    # Visualization code...

style_metric_cards()
)


