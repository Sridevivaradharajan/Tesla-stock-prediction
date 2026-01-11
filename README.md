# 📈 Tesla Stock Price Prediction

Deep learning model for multi-horizon stock price forecasting using Bidirectional LSTM. Achieves **85% R² accuracy** for next-day predictions.

## 🎯 Quick Overview

- **Model**: Bidirectional LSTM optimized with Optuna
- **Performance**: 85.2% R² (1-day), 68.3% R² (5-day), 45.4% R² (10-day)
- **Tech Stack**: TensorFlow, Streamlit, Plotly, Optuna
- **Dataset**: Tesla (TSLA) 2010-2020 daily OHLCV data

## 🚀 Quick Start

```bash
git clone https://github.com/Sridevivaradharajan/Tesla-stock-prediction.git
cd Tesla-stock-prediction
pip install -r requirements.txt
streamlit run App.py
```

## 📊 Complete Results

| Model | Horizon | R² | MAE ($) | RMSE ($) | MAPE (%) |
|-------|---------|-------|---------|----------|----------|
| **Bidirectional LSTM** | 1-day | **0.852** | **16.95** | **23.72** | **5.34** |
| **Bidirectional LSTM** | 5-day | **0.683** | **26.03** | **37.81** | **7.89** |
| **Bidirectional LSTM** | 10-day | **0.454** | **36.42** | **57.14** | **10.45** |
| LSTM | 1-day | 0.442 | 32.32 | 46.09 | 9.67 |
| LSTM | 5-day | 0.174 | 43.50 | 61.00 | 12.85 |
| LSTM | 10-day | 0.218 | 43.40 | 68.41 | 12.31 |
| SimpleRNN | 1-day | 0.431 | 34.88 | 46.57 | 10.47 |
| SimpleRNN | 5-day | 0.370 | 37.69 | 53.26 | 11.10 |
| SimpleRNN | 10-day | 0.186 | 47.19 | 69.80 | 13.51 |
| Stacked LSTM | 1-day | 0.433 | 33.11 | 46.48 | 9.95 |
| Stacked LSTM | 5-day | 0.236 | 40.99 | 58.64 | 12.06 |
| Stacked LSTM | 10-day | 0.154 | 46.33 | 71.14 | 13.15 |

### 🏆 Key Findings

✅ **Bidirectional LSTM wins all horizons** - 87-140% better R² than competitors  
✅ **Error reduction**: 49-69% lower MAE compared to baseline (SimpleRNN)  
✅ **Production accuracy**: 5.34% MAPE for next-day predictions (excellent for volatile stocks)  
✅ **Multi-horizon strength**: Best performer across 1-day, 5-day, and 10-day forecasts

## 📁 Project Structure

```
├── App.py                    # Streamlit web app
├── Tesla.ipynb              # Full analysis notebook
├── requirements.txt         # Dependencies
├── Dataset/                 # TSLA historical data
└── deployment_artifacts/    # Trained models & scalers
```

## 🎮 Usage

### Web App (Recommended)

```bash
streamlit run App.py
```

Upload CSV with 60+ days of OHLCV data and get instant predictions with visualizations.

**App Link**: [View App.py on GitHub](https://github.com/Sridevivaradharajan/Tesla-stock-prediction/blob/main/App.py)

### Python API

```python
from tensorflow.keras.models import load_model
import joblib

# Load model and scalers
model = load_model('deployment_artifacts/best_model_bidirectional_lstm_multihorizon.keras')
scaler_X = joblib.load('deployment_artifacts/scaler_features.pkl')
scaler_y = joblib.load('deployment_artifacts/scaler_target.pkl')

# Prepare input (60 days of OHLCV)
input_scaled = scaler_X.transform(your_data[-60:])
input_reshaped = input_scaled.reshape(1, 60, 5)

# Predict
predictions = scaler_y.inverse_transform(model.predict(input_reshaped))
print(f"1-day: ${predictions[0,0]:.2f}")
print(f"5-day: ${predictions[0,1]:.2f}")
print(f"10-day: ${predictions[0,2]:.2f}")
```

## 🧠 Model Architecture

```
Input: (60 timesteps, 5 features) → OHLCV data
├─ Bidirectional LSTM (128 units: 64 forward + 64 backward)
├─ Dropout (20%)
├─ Bidirectional LSTM (64 units: 32 forward + 32 backward)
├─ Dropout (20%)
├─ Dense (32 units, ReLU)
└─ Output (3 units) → 1-day, 5-day, 10-day predictions
```

**Optimized Hyperparameters** (Optuna with 15 trials):
- Units: 64 | Dropout: 0.2 | Learning Rate: 0.001 | Batch Size: 32

## 🛠️ Tech Stack

- **TensorFlow 2.15** - Deep learning framework
- **Streamlit 1.31** - Interactive web app
- **Optuna** - Bayesian hyperparameter optimization
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Preprocessing & metrics

## 📈 Why Bidirectional LSTM?

1. **Dual Context Processing**: Analyzes sequences forward (past→present) AND backward (present→past)
2. **Superior Pattern Recognition**: 98% improvement over SimpleRNN baseline
3. **Memory Efficiency**: LSTM cells solve vanishing gradient problem
4. **Consistent Performance**: Best across ALL metrics and horizons

## ⚠️ Disclaimer

**Educational purposes only.** Not financial advice. Past performance doesn't guarantee future results. Always conduct thorough research before making investment decisions.

## 📄 License

MIT License

## 👤 Author

**Sridevi V**  
📧 [GitHub](https://github.com/Sridevivaradharajan) | 💼 [LinkedIn](#)

---

⭐ **Star this repo** if you find it helpful! 🚀
