# ======================== LIBRRIES ===============================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import log_loss, confusion_matrix  # Import these
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss, confusion_matrix, accuracy_score # Added metrics

# ======================== TITLE & SIDEBAR ========================

# Streamlit Page Config
st.set_page_config(page_title="📈 Stock price predictor", layout="wide")
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .small-font {
        font-size:14px !important;
    }
    .key-data-box {
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# Side bar
st.sidebar.header("Inputs")
## User Input for Stock Symbol

s_ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT):", "AAPL").upper()

## User Input for Index Symbol
i_ticker = st.sidebar.text_input("Enter Index Symbol (e.g., ^GSPC for S&P500):", "^GSPC").upper()

## User input for risk-free rate
### Define risk-free rate options
risk_free_options = {
    "3-Month T-Bill (^IRX)": "^IRX",# 13 weeks yield T_bill
    "2-Year Yield Futures (^2YY=F)": "^2YY=F", # 2 years yield
    "5-Year T-Bill (^FVX)": "^FVX", # 5 years yield T_bill
    "10-Year COBE T-Bill (^TNX)": "^TNX", # CBOE 10 years yield T_bill (Chicago Board Options Exchange)
    "2-Year Treasury (^TYX)": "^TYX", # 30 years yield T_bill
}

### User selection
rf_ticker = st.sidebar.selectbox ("Select Risk-Free Rate", list(risk_free_options.keys()), index=0)


## User input for prediction days
prediction_days = st.sidebar.slider("Number of days to predict", 1, 7, 30)

# Title
st.markdown(f"<h1>📊 Stock Price Predictor <span style='color: blue;'>({s_ticker})</span></h1>", unsafe_allow_html=True)