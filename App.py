import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.metrics import MeanSquaredError
import traceback

# Page configuration
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1a1a2e;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        margin: 12px 0;
        border: 1px solid #e8edf2;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 16px;
        margin: 12px 0;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 16px;
        margin: 12px 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 16px;
        margin: 12px 0;
        border-radius: 4px;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 16px;
        margin: 12px 0;
        border-radius: 4px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 12px 28px;
        border: none;
        box-shadow: 0 2px 4px rgba(52, 152, 219, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
        transform: translateY(-1px);
    }
    .prediction-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        text-align: center;
        border-left: 4px solid;
        transition: transform 0.2s ease;
    }
    .prediction-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

@st.cache_resource
def load_model_artifacts():
    """Load model and related artifacts with enhanced compatibility"""
    try:
        deployment_dir = "deployment_artifacts"
        
        if not os.path.exists(deployment_dir):
            st.error(f"Deployment directory '{deployment_dir}' not found!")
            return None, None, None, None
        
        files_in_dir = os.listdir(deployment_dir)
        
        # Find model file
        model_file = None
        for file in files_in_dir:
            if file.endswith('.keras'):
                model_file = file
                break
        
        if not model_file:
            for file in files_in_dir:
                if file.endswith('.h5'):
                    model_file = file
                    break
        
        if not model_file:
            st.error("No model file (.keras or .h5) found in deployment_artifacts/")
            return None, None, None, None
        
        model_path = os.path.join(deployment_dir, model_file)
        
        # Try multiple loading strategies
        model = None
        loading_methods = [
            ("Standard load_model", lambda: load_model(model_path, compile=False)),
            ("Load with custom objects", lambda: load_model(
                model_path,
                custom_objects={'mse': MeanSquaredError(), 'MeanSquaredError': MeanSquaredError},
                compile=False
            )),
            ("TF 2.x compatibility mode", lambda: tf.keras.models.load_model(model_path, compile=False)),
        ]
        
        for method_name, load_func in loading_methods:
            try:
                model = load_func()
                break
            except Exception as e:
                st.warning(f"✗ {method_name} failed: {str(e)[:100]}")
                continue
        
        if model is None:
            st.error("All loading methods failed!")
            with st.expander("Show full error details"):
                st.code(traceback.format_exc())
            return None, None, None, None
        
        # Recompile model with simple configuration
        try:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        except Exception as e:
            st.warning(f"Recompilation warning: {str(e)}")
        
        # Load Scalers
        scaler_X_path = os.path.join(deployment_dir, 'scaler_features.pkl')
        scaler_y_path = os.path.join(deployment_dir, 'scaler_target.pkl')
        
        if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
            st.error("Scaler files not found!")
            return None, None, None, None
        
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        
        # Load Feature Columns
        feature_cols_path = os.path.join(deployment_dir, 'feature_columns.pkl')
        
        if os.path.exists(feature_cols_path):
            feature_columns = joblib.load(feature_cols_path)
        else:
            st.warning("Feature columns file not found, using default OHLCV")
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Load Configuration
        config_path = os.path.join(deployment_dir, 'model_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config_pkl_path = os.path.join(deployment_dir, 'model_config.pkl')
            if os.path.exists(config_pkl_path):
                config = joblib.load(config_pkl_path)
            else:
                st.warning("Config not found, using defaults")
                config = {}
        
        # Set default values
        config.setdefault('feature_columns', feature_columns)
        config.setdefault('lookback_window', 60)
        config.setdefault('model_name', 'Bidirectional LSTM')
        config.setdefault('framework', 'TensorFlow/Keras')
        config.setdefault('creation_date', '2026-01-10')
        config.setdefault('training_samples', 1873)
        config.setdefault('testing_samples', 423)
        
        if 'performance_metrics' not in config:
            config['performance_metrics'] = {
                '1-day': {'R2': 0.852279, 'RMSE': 23.720594, 'MAE': 16.949551, 'MAPE': 5.338108},
                '5-day': {'R2': 0.682578, 'RMSE': 37.812080, 'MAE': 26.025596, 'MAPE': 7.890270},
                '10-day': {'R2': 0.454400, 'RMSE': 57.136029, 'MAE': 36.421908, 'MAPE': 10.449523}
            }
        
        if 'best_hyperparameters' not in config:
            config['best_hyperparameters'] = {
                'units': 64,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        
        return model, scaler_X, scaler_y, config
        
    except Exception as e:
        st.error(f"Critical error loading artifacts: {str(e)}")
        with st.expander("View detailed error traceback"):
            st.code(traceback.format_exc())
        return None, None, None, None

def validate_input_data(df):
    """Validate uploaded data"""
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if not all(col in df.columns for col in required_columns):
        return False, f"Missing columns. Required: {', '.join(required_columns)}"
    
    if len(df) < 60:
        return False, f"Need at least 60 rows, got {len(df)}"
    
    for col in required_columns:
        if df[col].isnull().any():
            return False, f"Column '{col}' has missing values"
    
    return True, "Data validation successful"

def make_prediction(data, model, scaler_X, scaler_y, config):
    """Generate predictions using the trained model"""
    try:
        lookback = config.get('lookback_window', 60)
        feature_columns = config.get('feature_columns', ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Extract required features
        X = data[feature_columns].values
        
        if len(X) < lookback:
            st.error(f"Need at least {lookback} rows, got {len(X)}")
            return None
        
        # Take last lookback sequence
        input_data = X[-lookback:]
        
        # Scale
        input_scaled = scaler_X.transform(input_data)
        
        # Reshape for LSTM
        input_reshaped = input_scaled.reshape(1, lookback, len(feature_columns))
        
        # Predict
        prediction_scaled = model.predict(input_reshaped, verbose=0)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        
        return {
            '1_day_ahead': float(prediction[0, 0]),
            '5_day_ahead': float(prediction[0, 1]),
            '10_day_ahead': float(prediction[0, 2])
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        with st.expander("View detailed error"):
            st.code(traceback.format_exc())
        return None

def create_prediction_chart(historical_data, predictions, current_price):
    """Create interactive prediction visualization"""
    fig = go.Figure()
    
    historical_subset = historical_data.tail(60).copy()
    
    if 'Date' in historical_subset.columns:
        x_hist = pd.to_datetime(historical_subset['Date'])
        last_date = x_hist.iloc[-1]
    else:
        last_date = datetime.now()
        x_hist = pd.date_range(end=last_date, periods=len(historical_subset), freq='D')
    
    fig.add_trace(go.Scatter(
        x=x_hist,
        y=historical_subset['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#2c3e50', width=2.5),
        hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
    ))
    
    future_dates = [
        last_date + timedelta(days=1),
        last_date + timedelta(days=7),
        last_date + timedelta(days=14)
    ]
    
    predicted_prices = [
        predictions['1_day_ahead'],
        predictions['5_day_ahead'],
        predictions['10_day_ahead']
    ]
    
    fig.add_trace(go.Scatter(
        x=[last_date] + future_dates,
        y=[current_price] + predicted_prices,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#e74c3c', width=2.5, dash='dash'),
        marker=dict(size=10, symbol='diamond', line=dict(width=2, color='white')),
        hovertemplate='<b>Date</b>: %{x}<br><b>Predicted</b>: $%{y:.2f}<extra></extra>'
    ))
    
    volatility = historical_subset['Close'].std()
    upper_bound = [p + volatility * 1.96 for p in predicted_prices]
    lower_bound = [p - volatility * 1.96 for p in predicted_prices]
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper_bound,
        mode='lines',
        name='Upper Bound (95%)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower_bound,
        mode='lines',
        name='Confidence Interval',
        fill='tonexty',
        fillcolor='rgba(231, 76, 60, 0.15)',
        line=dict(width=0),
        showlegend=True,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title={
            'text': 'Multi-Horizon Stock Price Prediction',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1a1a2e'}
        },
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white',
        height=550,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='#fafafa'
    )
    
    return fig

def generate_report(predictions, current_price, config):
    """Generate downloadable report"""
    report = f"""
TESLA STOCK PRICE PREDICTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 70}

MODEL INFORMATION
Model: {config['model_name']}
Framework: {config['framework']}
Training Date: {config['creation_date']}
Lookback Window: {config['lookback_window']} days
Features Used: {', '.join(config['feature_columns'])}

CURRENT MARKET DATA
Current Price: ${current_price:.2f}
Data Source: User Uploaded CSV

PREDICTIONS
{'=' * 70}
Horizon     Predicted Price    Change ($)    Change (%)    Direction
{'=' * 70}
1-Day       ${predictions['1_day_ahead']:8.2f}      ${predictions['1_day_ahead'] - current_price:+8.2f}      {((predictions['1_day_ahead'] - current_price) / current_price * 100):+6.2f}%      {'BULLISH' if predictions['1_day_ahead'] > current_price else 'BEARISH'}
5-Day       ${predictions['5_day_ahead']:8.2f}      ${predictions['5_day_ahead'] - current_price:+8.2f}      {((predictions['5_day_ahead'] - current_price) / current_price * 100):+6.2f}%      {'BULLISH' if predictions['5_day_ahead'] > current_price else 'BEARISH'}
10-Day      ${predictions['10_day_ahead']:8.2f}      ${predictions['10_day_ahead'] - current_price:+8.2f}      {((predictions['10_day_ahead'] - current_price) / current_price * 100):+6.2f}%      {'BULLISH' if predictions['10_day_ahead'] > current_price else 'BEARISH'}
{'=' * 70}

MODEL PERFORMANCE METRICS
{'=' * 70}
Metric          1-Day      5-Day      10-Day
{'=' * 70}
R² Score        {config['performance_metrics']['1-day']['R2']:6.4f}     {config['performance_metrics']['5-day']['R2']:6.4f}     {config['performance_metrics']['10-day']['R2']:6.4f}
RMSE            ${config['performance_metrics']['1-day']['RMSE']:6.2f}     ${config['performance_metrics']['5-day']['RMSE']:6.2f}     ${config['performance_metrics']['10-day']['RMSE']:6.2f}
MAE             ${config['performance_metrics']['1-day']['MAE']:6.2f}     ${config['performance_metrics']['5-day']['MAE']:6.2f}     ${config['performance_metrics']['10-day']['MAE']:6.2f}
MAPE            {config['performance_metrics']['1-day']['MAPE']:5.2f}%      {config['performance_metrics']['5-day']['MAPE']:5.2f}%      {config['performance_metrics']['10-day']['MAPE']:5.2f}%
{'=' * 70}

DISCLAIMER
This prediction is generated by a machine learning model for educational
and research purposes only. It should NOT be considered as financial advice.

Report Generated By: Tesla Stock Price Prediction System
Developer: Sridevi V
Project: Deep Learning for Financial Time Series Forecasting
"""
    return report

# Main Application
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #2c3e50; margin: 10px 0;">Tesla Stock Predictor</h2>
            <p style="color: #7f8c8d;">Deep Learning Forecasting</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Prediction", "Model Comparison"],
            label_visibility="collapsed"
        )
    
    # Load model artifacts
    model, scaler_X, scaler_y, config = load_model_artifacts()
    
    if model is None:
        st.markdown("""
        <div class="error-box">
        <h3>Model Loading Failed</h3>
        <p>Please ensure the following files exist in the <code>deployment_artifacts/</code> folder:</p>
        <ul>
            <li>Model file (.keras or .h5)</li>
            <li>scaler_features.pkl</li>
            <li>scaler_target.pkl</li>
            <li>feature_columns.pkl (optional)</li>
            <li>model_config.json or model_config.pkl (optional)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main content based on page selection
    if page == "Prediction":
        st.title("Tesla Stock Price Prediction")
        st.markdown("Upload historical stock data to generate multi-horizon price predictions")
        
        uploaded_file = st.file_uploader(
            "Upload Stock Data (CSV)",
            type=['csv'],
            help="Upload CSV with at least 60 days of OHLCV data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Clean and convert numeric columns
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_columns:
                    if col in df.columns:
                        # Remove $ signs and convert to float
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.replace('
                
                is_valid, message = validate_input_data(df)
                
                if is_valid:
                    st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", f"{len(df):,}")
                    with col2:
                        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                    with col3:
                        if 'Date' in df.columns:
                            st.metric("Start Date", df['Date'].min().strftime('%Y-%m-%d'))
                        else:
                            st.metric("Min Price", f"${df['Close'].min():.2f}")
                    with col4:
                        if 'Date' in df.columns:
                            st.metric("End Date", df['Date'].max().strftime('%Y-%m-%d'))
                        else:
                            st.metric("Max Price", f"${df['Close'].max():.2f}")
                    
                    with st.expander("View Data Preview"):
                        st.dataframe(df.tail(15), use_container_width=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Generate Predictions", type="primary", use_container_width=True):
                        with st.spinner("Analyzing patterns and generating predictions..."):
                            predictions = make_prediction(df, model, scaler_X, scaler_y, config)
                            
                            if predictions:
                                st.session_state.predictions = predictions
                                current_price = df['Close'].iloc[-1]
                                
                                st.markdown("---")
                                st.subheader("Multi-Horizon Predictions")
                                
                                change_1d = ((predictions['1_day_ahead'] - current_price) / current_price * 100)
                                change_5d = ((predictions['5_day_ahead'] - current_price) / current_price * 100)
                                change_10d = ((predictions['10_day_ahead'] - current_price) / current_price * 100)
                                
                                if 'Date' in df.columns:
                                    last_date = pd.to_datetime(df['Date'].iloc[-1])
                                else:
                                    last_date = datetime.now()
                                
                                pred_date_1d = last_date + timedelta(days=1)
                                pred_date_5d = last_date + timedelta(days=7)
                                pred_date_10d = last_date + timedelta(days=14)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="prediction-card" style="border-left-color: #2c3e50;">
                                        <h4 style="margin: 0; color: #666; font-size: 14px;">1-Day Forecast</h4>
                                        <p style="margin: 5px 0; color: #999; font-size: 12px;">{pred_date_1d.strftime('%Y-%m-%d')}</p>
                                        <h2 style="margin: 10px 0; color: #2c3e50; font-size: 32px;">${predictions['1_day_ahead']:.2f}</h2>
                                        <p style="margin: 0; color: {'#27ae60' if change_1d > 0 else '#e74c3c'}; font-weight: 600; font-size: 18px;">
                                            {change_1d:+.2f}%
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="prediction-card" style="border-left-color: #3498db;">
                                        <h4 style="margin: 0; color: #666; font-size: 14px;">5-Day Forecast</h4>
                                        <p style="margin: 5px 0; color: #999; font-size: 12px;">{pred_date_5d.strftime('%Y-%m-%d')}</p>
                                        <h2 style="margin: 10px 0; color: #3498db; font-size: 32px;">${predictions['5_day_ahead']:.2f}</h2>
                                        <p style="margin: 0; color: {'#27ae60' if change_5d > 0 else '#e74c3c'}; font-weight: 600; font-size: 18px;">
                                            {change_5d:+.2f}%
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown(f"""
                                    <div class="prediction-card" style="border-left-color: #2980b9;">
                                        <h4 style="margin: 0; color: #666; font-size: 14px;">10-Day Forecast</h4>
                                        <p style="margin: 5px 0; color: #999; font-size: 12px;">{pred_date_10d.strftime('%Y-%m-%d')}</p>
                                        <h2 style="margin: 10px 0; color: #2980b9; font-size: 32px;">${predictions['10_day_ahead']:.2f}</h2>
                                        <p style="margin: 0; color: {'#27ae60' if change_10d > 0 else '#e74c3c'}; font-weight: 600; font-size: 18px;">
                                            {change_10d:+.2f}%
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown("### Detailed Analysis")
                                
                                # Store raw numeric values, not formatted strings
                                predictions_df = pd.DataFrame({
                                    'Horizon': ['1-Day Ahead', '5-Day Ahead', '10-Day Ahead'],
                                    'Target Date': [
                                        pred_date_1d.strftime('%Y-%m-%d'),
                                        pred_date_5d.strftime('%Y-%m-%d'),
                                        pred_date_10d.strftime('%Y-%m-%d')
                                    ],
                                    'Predicted Price': [
                                        predictions["1_day_ahead"],
                                        predictions["5_day_ahead"],
                                        predictions["10_day_ahead"]
                                    ],
                                    'Expected Change': [
                                        predictions["1_day_ahead"] - current_price,
                                        predictions["5_day_ahead"] - current_price,
                                        predictions["10_day_ahead"] - current_price
                                    ],
                                    'Change %': [
                                        change_1d,
                                        change_5d,
                                        change_10d
                                    ],
                                    'Signal': [
                                        'BULLISH' if change_1d > 0 else 'BEARISH',
                                        'BULLISH' if change_5d > 0 else 'BEARISH',
                                        'BULLISH' if change_10d > 0 else 'BEARISH'
                                    ]
                                })
                                
                                # Format display using Styler
                                styled_df = predictions_df.style.format({
                                    'Predicted Price': '${:.2f}',
                                    'Expected Change': '${:+.2f}',
                                    'Change %': '{:+.2f}%'
                                })
                                
                                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                                
                                st.markdown("---")
                                st.markdown("### Price Prediction Visualization")
                                fig = create_prediction_chart(df, predictions, current_price)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("---")
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown("""
                                    <div class="warning-box">
                                    <strong>Disclaimer:</strong> These predictions are generated by a machine learning model 
                                    and should not be considered as financial advice. Always conduct thorough research 
                                    before making investment decisions.
                                    </div>
                                    """, unsafe_allow_html=True)
                                with col2:
                                    report = generate_report(predictions, current_price, config)
                                    st.download_button(
                                        label="Download Report",
                                        data=report,
                                        file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                        mime="text/plain",
                                        use_container_width=True
                                    )
                
                else:
                    st.markdown(f'<div class="warning-box">{message}</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                with st.expander("View detailed error"):
                    st.code(traceback.format_exc())
        
        else:
            st.markdown("""
            <div class="info-box">
            <strong>Getting Started:</strong><br>
            Upload a CSV file containing Tesla stock data with:
            <ul>
                <li>At least 60 days of historical data</li>
                <li>Required columns: Open, High, Low, Close, Volume</li>
                <li>Optional: Date column for better visualization</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "Model Comparison":
        st.title("Model Comparison: RNN Architectures")
        st.markdown("Comprehensive comparison of different RNN architectures for stock prediction")
        
        # Deployed Model Details
        st.subheader("Deployed Model Specifications")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Architecture Details</h4>
                <p><strong>Model:</strong> {config['model_name']}</p>
                <p><strong>Framework:</strong> {config['framework']}</p>
                <p><strong>Lookback Window:</strong> {config['lookback_window']} days</p>
                <p><strong>Features:</strong> {', '.join(config['feature_columns'])}</p>
                <p><strong>Training Samples:</strong> {config['training_samples']:,}</p>
                <p><strong>Testing Samples:</strong> {config['testing_samples']:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Best Hyperparameters (Optuna)</h4>
                <p><strong>Units:</strong> {config['best_hyperparameters']['units']}</p>
                <p><strong>Dropout:</strong> {config['best_hyperparameters']['dropout']}</p>
                <p><strong>Learning Rate:</strong> {config['best_hyperparameters']['learning_rate']}</p>
                <p><strong>Batch Size:</strong> {config['best_hyperparameters']['batch_size']}</p>
                <p><strong>Optimization:</strong> Bayesian (TPE Sampler)</p>
                <p><strong>Creation Date:</strong> {config['creation_date']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparison data
        comparison_data = {
            'Model': ['SimpleRNN', 'SimpleRNN', 'SimpleRNN', 
                     'LSTM', 'LSTM', 'LSTM',
                     'Bidirectional LSTM', 'Bidirectional LSTM', 'Bidirectional LSTM',
                     'Stacked LSTM', 'Stacked LSTM', 'Stacked LSTM'],
            'Horizon': ['1-day', '5-day', '10-day'] * 4,
            'MAE': [34.88, 37.69, 47.19, 32.32, 43.50, 43.39, 16.95, 26.03, 36.42, 33.11, 40.99, 46.33],
            'RMSE': [46.57, 53.26, 69.80, 46.09, 61.00, 68.41, 23.72, 37.81, 57.14, 46.48, 58.64, 71.14],
            'R²': [0.431, 0.370, 0.186, 0.442, 0.174, 0.218, 0.852, 0.683, 0.454, 0.433, 0.236, 0.154],
            'MAPE': [10.47, 11.10, 13.51, 9.67, 12.85, 12.31, 5.34, 7.89, 10.45, 9.95, 12.06, 13.15]
        }
        
        comp_df = pd.DataFrame(comparison_data)
        
        st.subheader("Performance Comparison")
        
        tab1, tab2, tab3 = st.tabs(["By Horizon", "By Model", "Full Table"])
        
        with tab1:
            horizon_select = st.selectbox("Select Prediction Horizon", ['1-day', '5-day', '10-day'])
            horizon_data = comp_df[comp_df['Horizon'] == horizon_select].copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_r2 = go.Figure(data=[
                    go.Bar(x=horizon_data['Model'], y=horizon_data['R²'], 
                          marker_color=['#e74c3c', '#f39c12', '#27ae60', '#3498db'],
                          text=horizon_data['R²'].round(3), textposition='auto')
                ])
                fig_r2.update_layout(title=f'R² Score Comparison ({horizon_select})',
                                    yaxis_title='R² Score', height=400, template='plotly_white')
                st.plotly_chart(fig_r2, use_container_width=True)
                
                fig_mae = go.Figure(data=[
                    go.Bar(x=horizon_data['Model'], y=horizon_data['MAE'],
                          marker_color=['#e74c3c', '#f39c12', '#27ae60', '#3498db'],
                          text=horizon_data['MAE'].round(2), textposition='auto')
                ])
                fig_mae.update_layout(title=f'MAE Comparison ({horizon_select})',
                                     yaxis_title='MAE ($)', height=400, template='plotly_white')
                st.plotly_chart(fig_mae, use_container_width=True)
            
            with col2:
                fig_rmse = go.Figure(data=[
                    go.Bar(x=horizon_data['Model'], y=horizon_data['RMSE'],
                          marker_color=['#e74c3c', '#f39c12', '#27ae60', '#3498db'],
                          text=horizon_data['RMSE'].round(2), textposition='auto')
                ])
                fig_rmse.update_layout(title=f'RMSE Comparison ({horizon_select})',
                                      yaxis_title='RMSE ($)', height=400, template='plotly_white')
                st.plotly_chart(fig_rmse, use_container_width=True)
                
                fig_mape = go.Figure(data=[
                    go.Bar(x=horizon_data['Model'], y=horizon_data['MAPE'],
                          marker_color=['#e74c3c', '#f39c12', '#27ae60', '#3498db'],
                          text=horizon_data['MAPE'].round(2), textposition='auto')
                ])
                fig_mape.update_layout(title=f'MAPE Comparison ({horizon_select})',
                                      yaxis_title='MAPE (%)', height=400, template='plotly_white')
                st.plotly_chart(fig_mape, use_container_width=True)
        
        with tab2:
            model_select = st.selectbox("Select Model", ['SimpleRNN', 'LSTM', 'Bidirectional LSTM', 'Stacked LSTM'])
            model_data = comp_df[comp_df['Model'] == model_select].copy()
            
            fig_multi = make_subplots(
                rows=2, cols=2,
                subplot_titles=('R² Score', 'RMSE ($)', 'MAE ($)', 'MAPE (%)'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            colors_horizon = ['#2c3e50', '#3498db', '#2980b9']
            
            fig_multi.add_trace(go.Bar(x=model_data['Horizon'], y=model_data['R²'], 
                                       marker_color=colors_horizon, showlegend=False), row=1, col=1)
            fig_multi.add_trace(go.Bar(x=model_data['Horizon'], y=model_data['RMSE'],
                                       marker_color=colors_horizon, showlegend=False), row=1, col=2)
            fig_multi.add_trace(go.Bar(x=model_data['Horizon'], y=model_data['MAE'],
                                       marker_color=colors_horizon, showlegend=False), row=2, col=1)
            fig_multi.add_trace(go.Bar(x=model_data['Horizon'], y=model_data['MAPE'],
                                       marker_color=colors_horizon, showlegend=False), row=2, col=2)
            
            fig_multi.update_layout(height=600, title_text=f"{model_select} Performance Across Horizons",
                                   template='plotly_white')
            st.plotly_chart(fig_multi, use_container_width=True)
        
        with tab3:
            st.dataframe(comp_df.style.format({
                'MAE': '{:.2f}',
                'RMSE': '{:.2f}',
                'R²': '{:.4f}',
                'MAPE': '{:.2f}'
            }), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("Model Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>Strengths of Bidirectional LSTM</h4>
            <ul>
                <li>High R² score (0.85) for 1-day predictions</li>
                <li>Low MAPE (5.34%) indicating precise forecasts</li>
                <li>Bidirectional processing captures temporal patterns</li>
                <li>60-day lookback provides rich context</li>
                <li>Trained on 1,873 samples (2010-2019 data)</li>
                <li>2-3x more accurate than SimpleRNN</li>
                <li>Optimized using Optuna (15 trials, Bayesian search)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
            <h4>Limitations & Considerations</h4>
            <ul>
                <li>Performance decreases for longer horizons</li>
                <li>Does not account for external events</li>
                <li>Requires minimum 60 days of data</li>
                <li>Market volatility affects accuracy</li>
                <li>Not suitable for extreme market conditions</li>
                <li>Black swan events cannot be predicted</li>
                <li>R² declines: 0.85 (1d) → 0.68 (5d) → 0.45 (10d)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Why Bidirectional LSTM Won")
        
        st.markdown("""
        <div class="success-box">
        <h4>Comparative Analysis</h4>
        
        <p><strong>1. Bidirectional LSTM (Selected Model) - Best Overall</strong></p>
        <ul>
            <li>Highest R² (0.852 for 1-day) - explains 85% of variance</li>
            <li>Lowest MAE ($16.95 for 1-day) - most accurate predictions</li>
            <li>Lowest RMSE ($23.72 for 1-day) - smallest prediction errors</li>
            <li>Lowest MAPE (5.34% for 1-day) - best percentage accuracy</li>
            <li><strong>Key Advantage:</strong> Processes sequences bidirectionally (forward + backward)</li>
            <li><strong>Hyperparameters:</strong> 64 units, 0.2 dropout, 0.001 learning rate</li>
        </ul>
        
        <p><strong>2. LSTM - Moderate Performance</strong></p>
        <ul>
            <li>R² ranges from 0.17 to 0.44 across horizons</li>
            <li>Better than SimpleRNN but lacks bidirectional context</li>
            <li>Unidirectional processing limits pattern recognition</li>
        </ul>
        
        <p><strong>3. Stacked LSTM - Inconsistent</strong></p>
        <ul>
            <li>Similar to LSTM performance (R² 0.15-0.43)</li>
            <li>Deeper architecture doesn't improve results</li>
            <li>May suffer from overfitting or gradient issues</li>
        </ul>
        
        <p><strong>4. SimpleRNN - Weakest Performance</strong></p>
        <ul>
            <li>Lowest R² scores (0.19 to 0.43)</li>
            <li>Highest errors: MAPE up to 13.51%</li>
            <li><strong>Why it fails:</strong> Cannot capture long-term dependencies</li>
            <li>Suffers from vanishing gradient problem</li>
        </ul>
        
        <p><strong>Optimization Process (Optuna):</strong></p>
        <ul>
            <li>Bayesian hyperparameter search (TPE Sampler)</li>
            <li>15 trials per model with early pruning</li>
            <li>Time-series cross-validation (2 splits)</li>
            <li>Median pruner for computational efficiency</li>
        </ul>
        
        <p><strong>Conclusion:</strong> Bidirectional LSTM achieves 98% improvement over SimpleRNN 
        and 93% improvement over standard LSTM for 1-day predictions, making it production-ready.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main(), '').str.replace(',', '')
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.sort_values('Date').reset_index(drop=True)
                
                st.session_state.uploaded_data = df
                
                is_valid, message = validate_input_data(df)
                
                if is_valid:
                    st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", f"{len(df):,}")
                    with col2:
                        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                    with col3:
                        if 'Date' in df.columns:
                            st.metric("Start Date", df['Date'].min().strftime('%Y-%m-%d'))
                        else:
                            st.metric("Min Price", f"${df['Close'].min():.2f}")
                    with col4:
                        if 'Date' in df.columns:
                            st.metric("End Date", df['Date'].max().strftime('%Y-%m-%d'))
                        else:
                            st.metric("Max Price", f"${df['Close'].max():.2f}")
                    
                    with st.expander("View Data Preview"):
                        st.dataframe(df.tail(15), use_container_width=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Generate Predictions", type="primary", use_container_width=True):
                        with st.spinner("Analyzing patterns and generating predictions..."):
                            predictions = make_prediction(df, model, scaler_X, scaler_y, config)
                            
                            if predictions:
                                st.session_state.predictions = predictions
                                current_price = df['Close'].iloc[-1]
                                
                                st.markdown("---")
                                st.subheader("Multi-Horizon Predictions")
                                
                                change_1d = ((predictions['1_day_ahead'] - current_price) / current_price * 100)
                                change_5d = ((predictions['5_day_ahead'] - current_price) / current_price * 100)
                                change_10d = ((predictions['10_day_ahead'] - current_price) / current_price * 100)
                                
                                if 'Date' in df.columns:
                                    last_date = pd.to_datetime(df['Date'].iloc[-1])
                                else:
                                    last_date = datetime.now()
                                
                                pred_date_1d = last_date + timedelta(days=1)
                                pred_date_5d = last_date + timedelta(days=7)
                                pred_date_10d = last_date + timedelta(days=14)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="prediction-card" style="border-left-color: #2c3e50;">
                                        <h4 style="margin: 0; color: #666; font-size: 14px;">1-Day Forecast</h4>
                                        <p style="margin: 5px 0; color: #999; font-size: 12px;">{pred_date_1d.strftime('%Y-%m-%d')}</p>
                                        <h2 style="margin: 10px 0; color: #2c3e50; font-size: 32px;">${predictions['1_day_ahead']:.2f}</h2>
                                        <p style="margin: 0; color: {'#27ae60' if change_1d > 0 else '#e74c3c'}; font-weight: 600; font-size: 18px;">
                                            {change_1d:+.2f}%
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="prediction-card" style="border-left-color: #3498db;">
                                        <h4 style="margin: 0; color: #666; font-size: 14px;">5-Day Forecast</h4>
                                        <p style="margin: 5px 0; color: #999; font-size: 12px;">{pred_date_5d.strftime('%Y-%m-%d')}</p>
                                        <h2 style="margin: 10px 0; color: #3498db; font-size: 32px;">${predictions['5_day_ahead']:.2f}</h2>
                                        <p style="margin: 0; color: {'#27ae60' if change_5d > 0 else '#e74c3c'}; font-weight: 600; font-size: 18px;">
                                            {change_5d:+.2f}%
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown(f"""
                                    <div class="prediction-card" style="border-left-color: #2980b9;">
                                        <h4 style="margin: 0; color: #666; font-size: 14px;">10-Day Forecast</h4>
                                        <p style="margin: 5px 0; color: #999; font-size: 12px;">{pred_date_10d.strftime('%Y-%m-%d')}</p>
                                        <h2 style="margin: 10px 0; color: #2980b9; font-size: 32px;">${predictions['10_day_ahead']:.2f}</h2>
                                        <p style="margin: 0; color: {'#27ae60' if change_10d > 0 else '#e74c3c'}; font-weight: 600; font-size: 18px;">
                                            {change_10d:+.2f}%
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown("### Detailed Analysis")
                                
                                # Store raw numeric values, not formatted strings
                                predictions_df = pd.DataFrame({
                                    'Horizon': ['1-Day Ahead', '5-Day Ahead', '10-Day Ahead'],
                                    'Target Date': [
                                        pred_date_1d.strftime('%Y-%m-%d'),
                                        pred_date_5d.strftime('%Y-%m-%d'),
                                        pred_date_10d.strftime('%Y-%m-%d')
                                    ],
                                    'Predicted Price': [
                                        predictions["1_day_ahead"],
                                        predictions["5_day_ahead"],
                                        predictions["10_day_ahead"]
                                    ],
                                    'Expected Change': [
                                        predictions["1_day_ahead"] - current_price,
                                        predictions["5_day_ahead"] - current_price,
                                        predictions["10_day_ahead"] - current_price
                                    ],
                                    'Change %': [
                                        change_1d,
                                        change_5d,
                                        change_10d
                                    ],
                                    'Signal': [
                                        'BULLISH' if change_1d > 0 else 'BEARISH',
                                        'BULLISH' if change_5d > 0 else 'BEARISH',
                                        'BULLISH' if change_10d > 0 else 'BEARISH'
                                    ]
                                })
                                
                                # Format display using Styler
                                styled_df = predictions_df.style.format({
                                    'Predicted Price': '${:.2f}',
                                    'Expected Change': '${:+.2f}',
                                    'Change %': '{:+.2f}%'
                                })
                                
                                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                                
                                st.markdown("---")
                                st.markdown("### Price Prediction Visualization")
                                fig = create_prediction_chart(df, predictions, current_price)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("---")
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown("""
                                    <div class="warning-box">
                                    <strong>Disclaimer:</strong> These predictions are generated by a machine learning model 
                                    and should not be considered as financial advice. Always conduct thorough research 
                                    before making investment decisions.
                                    </div>
                                    """, unsafe_allow_html=True)
                                with col2:
                                    report = generate_report(predictions, current_price, config)
                                    st.download_button(
                                        label="Download Report",
                                        data=report,
                                        file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                        mime="text/plain",
                                        use_container_width=True
                                    )
                
                else:
                    st.markdown(f'<div class="warning-box">{message}</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                with st.expander("View detailed error"):
                    st.code(traceback.format_exc())
        
        else:
            st.markdown("""
            <div class="info-box">
            <strong>Getting Started:</strong><br>
            Upload a CSV file containing Tesla stock data with:
            <ul>
                <li>At least 60 days of historical data</li>
                <li>Required columns: Open, High, Low, Close, Volume</li>
                <li>Optional: Date column for better visualization</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "Model Comparison":
        st.title("Model Comparison: RNN Architectures")
        st.markdown("Comprehensive comparison of different RNN architectures for stock prediction")
        
        # Deployed Model Details
        st.subheader("Deployed Model Specifications")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Architecture Details</h4>
                <p><strong>Model:</strong> {config['model_name']}</p>
                <p><strong>Framework:</strong> {config['framework']}</p>
                <p><strong>Lookback Window:</strong> {config['lookback_window']} days</p>
                <p><strong>Features:</strong> {', '.join(config['feature_columns'])}</p>
                <p><strong>Training Samples:</strong> {config['training_samples']:,}</p>
                <p><strong>Testing Samples:</strong> {config['testing_samples']:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Best Hyperparameters (Optuna)</h4>
                <p><strong>Units:</strong> {config['best_hyperparameters']['units']}</p>
                <p><strong>Dropout:</strong> {config['best_hyperparameters']['dropout']}</p>
                <p><strong>Learning Rate:</strong> {config['best_hyperparameters']['learning_rate']}</p>
                <p><strong>Batch Size:</strong> {config['best_hyperparameters']['batch_size']}</p>
                <p><strong>Optimization:</strong> Bayesian (TPE Sampler)</p>
                <p><strong>Creation Date:</strong> {config['creation_date']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparison data
        comparison_data = {
            'Model': ['SimpleRNN', 'SimpleRNN', 'SimpleRNN', 
                     'LSTM', 'LSTM', 'LSTM',
                     'Bidirectional LSTM', 'Bidirectional LSTM', 'Bidirectional LSTM',
                     'Stacked LSTM', 'Stacked LSTM', 'Stacked LSTM'],
            'Horizon': ['1-day', '5-day', '10-day'] * 4,
            'MAE': [34.88, 37.69, 47.19, 32.32, 43.50, 43.39, 16.95, 26.03, 36.42, 33.11, 40.99, 46.33],
            'RMSE': [46.57, 53.26, 69.80, 46.09, 61.00, 68.41, 23.72, 37.81, 57.14, 46.48, 58.64, 71.14],
            'R²': [0.431, 0.370, 0.186, 0.442, 0.174, 0.218, 0.852, 0.683, 0.454, 0.433, 0.236, 0.154],
            'MAPE': [10.47, 11.10, 13.51, 9.67, 12.85, 12.31, 5.34, 7.89, 10.45, 9.95, 12.06, 13.15]
        }
        
        comp_df = pd.DataFrame(comparison_data)
        
        st.subheader("Performance Comparison")
        
        tab1, tab2, tab3 = st.tabs(["By Horizon", "By Model", "Full Table"])
        
        with tab1:
            horizon_select = st.selectbox("Select Prediction Horizon", ['1-day', '5-day', '10-day'])
            horizon_data = comp_df[comp_df['Horizon'] == horizon_select].copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_r2 = go.Figure(data=[
                    go.Bar(x=horizon_data['Model'], y=horizon_data['R²'], 
                          marker_color=['#e74c3c', '#f39c12', '#27ae60', '#3498db'],
                          text=horizon_data['R²'].round(3), textposition='auto')
                ])
                fig_r2.update_layout(title=f'R² Score Comparison ({horizon_select})',
                                    yaxis_title='R² Score', height=400, template='plotly_white')
                st.plotly_chart(fig_r2, use_container_width=True)
                
                fig_mae = go.Figure(data=[
                    go.Bar(x=horizon_data['Model'], y=horizon_data['MAE'],
                          marker_color=['#e74c3c', '#f39c12', '#27ae60', '#3498db'],
                          text=horizon_data['MAE'].round(2), textposition='auto')
                ])
                fig_mae.update_layout(title=f'MAE Comparison ({horizon_select})',
                                     yaxis_title='MAE ($)', height=400, template='plotly_white')
                st.plotly_chart(fig_mae, use_container_width=True)
            
            with col2:
                fig_rmse = go.Figure(data=[
                    go.Bar(x=horizon_data['Model'], y=horizon_data['RMSE'],
                          marker_color=['#e74c3c', '#f39c12', '#27ae60', '#3498db'],
                          text=horizon_data['RMSE'].round(2), textposition='auto')
                ])
                fig_rmse.update_layout(title=f'RMSE Comparison ({horizon_select})',
                                      yaxis_title='RMSE ($)', height=400, template='plotly_white')
                st.plotly_chart(fig_rmse, use_container_width=True)
                
                fig_mape = go.Figure(data=[
                    go.Bar(x=horizon_data['Model'], y=horizon_data['MAPE'],
                          marker_color=['#e74c3c', '#f39c12', '#27ae60', '#3498db'],
                          text=horizon_data['MAPE'].round(2), textposition='auto')
                ])
                fig_mape.update_layout(title=f'MAPE Comparison ({horizon_select})',
                                      yaxis_title='MAPE (%)', height=400, template='plotly_white')
                st.plotly_chart(fig_mape, use_container_width=True)
        
        with tab2:
            model_select = st.selectbox("Select Model", ['SimpleRNN', 'LSTM', 'Bidirectional LSTM', 'Stacked LSTM'])
            model_data = comp_df[comp_df['Model'] == model_select].copy()
            
            fig_multi = make_subplots(
                rows=2, cols=2,
                subplot_titles=('R² Score', 'RMSE ($)', 'MAE ($)', 'MAPE (%)'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            colors_horizon = ['#2c3e50', '#3498db', '#2980b9']
            
            fig_multi.add_trace(go.Bar(x=model_data['Horizon'], y=model_data['R²'], 
                                       marker_color=colors_horizon, showlegend=False), row=1, col=1)
            fig_multi.add_trace(go.Bar(x=model_data['Horizon'], y=model_data['RMSE'],
                                       marker_color=colors_horizon, showlegend=False), row=1, col=2)
            fig_multi.add_trace(go.Bar(x=model_data['Horizon'], y=model_data['MAE'],
                                       marker_color=colors_horizon, showlegend=False), row=2, col=1)
            fig_multi.add_trace(go.Bar(x=model_data['Horizon'], y=model_data['MAPE'],
                                       marker_color=colors_horizon, showlegend=False), row=2, col=2)
            
            fig_multi.update_layout(height=600, title_text=f"{model_select} Performance Across Horizons",
                                   template='plotly_white')
            st.plotly_chart(fig_multi, use_container_width=True)
        
        with tab3:
            st.dataframe(comp_df.style.format({
                'MAE': '{:.2f}',
                'RMSE': '{:.2f}',
                'R²': '{:.4f}',
                'MAPE': '{:.2f}'
            }), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("Model Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>Strengths of Bidirectional LSTM</h4>
            <ul>
                <li>High R² score (0.85) for 1-day predictions</li>
                <li>Low MAPE (5.34%) indicating precise forecasts</li>
                <li>Bidirectional processing captures temporal patterns</li>
                <li>60-day lookback provides rich context</li>
                <li>Trained on 1,873 samples (2010-2019 data)</li>
                <li>2-3x more accurate than SimpleRNN</li>
                <li>Optimized using Optuna (15 trials, Bayesian search)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
            <h4>Limitations & Considerations</h4>
            <ul>
                <li>Performance decreases for longer horizons</li>
                <li>Does not account for external events</li>
                <li>Requires minimum 60 days of data</li>
                <li>Market volatility affects accuracy</li>
                <li>Not suitable for extreme market conditions</li>
                <li>Black swan events cannot be predicted</li>
                <li>R² declines: 0.85 (1d) → 0.68 (5d) → 0.45 (10d)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Why Bidirectional LSTM Won")
        
        st.markdown("""
        <div class="success-box">
        <h4>Comparative Analysis</h4>
        
        <p><strong>1. Bidirectional LSTM (Selected Model) - Best Overall</strong></p>
        <ul>
            <li>Highest R² (0.852 for 1-day) - explains 85% of variance</li>
            <li>Lowest MAE ($16.95 for 1-day) - most accurate predictions</li>
            <li>Lowest RMSE ($23.72 for 1-day) - smallest prediction errors</li>
            <li>Lowest MAPE (5.34% for 1-day) - best percentage accuracy</li>
            <li><strong>Key Advantage:</strong> Processes sequences bidirectionally (forward + backward)</li>
            <li><strong>Hyperparameters:</strong> 64 units, 0.2 dropout, 0.001 learning rate</li>
        </ul>
        
        <p><strong>2. LSTM - Moderate Performance</strong></p>
        <ul>
            <li>R² ranges from 0.17 to 0.44 across horizons</li>
            <li>Better than SimpleRNN but lacks bidirectional context</li>
            <li>Unidirectional processing limits pattern recognition</li>
        </ul>
        
        <p><strong>3. Stacked LSTM - Inconsistent</strong></p>
        <ul>
            <li>Similar to LSTM performance (R² 0.15-0.43)</li>
            <li>Deeper architecture doesn't improve results</li>
            <li>May suffer from overfitting or gradient issues</li>
        </ul>
        
        <p><strong>4. SimpleRNN - Weakest Performance</strong></p>
        <ul>
            <li>Lowest R² scores (0.19 to 0.43)</li>
            <li>Highest errors: MAPE up to 13.51%</li>
            <li><strong>Why it fails:</strong> Cannot capture long-term dependencies</li>
            <li>Suffers from vanishing gradient problem</li>
        </ul>
        
        <p><strong>Optimization Process (Optuna):</strong></p>
        <ul>
            <li>Bayesian hyperparameter search (TPE Sampler)</li>
            <li>15 trials per model with early pruning</li>
            <li>Time-series cross-validation (2 splits)</li>
            <li>Median pruner for computational efficiency</li>
        </ul>
        
        <p><strong>Conclusion:</strong> Bidirectional LSTM achieves 98% improvement over SimpleRNN 
        and 93% improvement over standard LSTM for 1-day predictions, making it production-ready.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
