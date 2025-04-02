# =====================Stock price prediction=======================#
# This project intents to create a python program to predict stock prices using three 
# different models: LSTM, Randomforest, Regression.
# The program also calculates the Expected Retrun of the selected market and shows
# Visualisation of the real time asset price and some indicators such as 50Ma, 200Ma
# RSI.
# The program also provides AI based interpretation of the asset situation and analysis of the
# Machine learning models.

# ======================== Step 1: LIBRRIES ===============================

import streamlit as st
from streamlit_float import *
import yfinance as yf
import pandas as pd
import numpy as np
import inspect
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from adjustText import adjust_text
from streamlit_autorefresh import st_autorefresh
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import datetime, timedelta
# import matplotlib.ticker as ticker
# from matplotlib.ticker import PercentFormatter

# import pytz

# import matplotlib.pyplot as plt
# from scipy import stats
# from sklearn.metrics import log_loss, confusion_matrix  # Import these
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss, confusion_matrix, accuracy_score # Added metrics

# ======================Step 2: Page configuration & SIDEBAR ========================

## Streamlit Configuriation and CSS

### Page Configuration
st.set_page_config(
    page_title="📈 Stock Price Predictor",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

### CSS 
# Define custom CSS for full-page background color
page_bg_color = """
<style>
    body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #D2B48C;  /* Light Brown */
    }
    [data-testid="stToolbar"] {
        display: none;  /* Hides the Streamlit toolbar */
    }
    .stApp {
        background-color: #D2B48C !important;
    }
</style>
"""
# Inject the custom CSS
st.markdown(page_bg_color, unsafe_allow_html=True)

# Define custom CSS for fonts

st.markdown("""
    <style>
    # Font styles
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
m_ticker = st.sidebar.text_input("Enter Index Symbol (e.g., ^GSPC for S&P500):", "^GSPC").upper()

## User input for prediction days
prediction_days = st.sidebar.slider("Number of days to predict", 1, 7, 30)

# ======================== STEP 3: DEFINING FUNCTIONS ========================

## Function to get Historical Data
@st.cache_data
def load_data(stock_symbol, market_symbol, rf_symbol, period):
    text_title ="LOADING DATA"
    print(f"{'=' * 25} {text_title} {'=' * 25}")
    text_length = len(text_title)+2

    symbols = [stock_symbol, market_symbol, rf_symbol]
    stock_history, market_history, rf_history = None, None, None

    for i, symbol in enumerate(symbols):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if df.empty:
                print(f"No data available for {symbol}")
            else:
                print(f"{symbol} data fetched: OK")
                
                if i == 0:
                    stock_history = df
                    
                elif i == 1:
                    market_history = df
                else:
                    rf_history = df
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")

    print('=' * (50 + text_length))

    return stock_history, market_history, rf_history

## Function to get live Stock data


## Function to localize index(Date) column
def localize(stock_history, market_history, rf_history):

    text_title ="LOCALIZE INDEX"
    print(f"{'=' * 25} {text_title} {'=' * 25}")
    text_length = len(text_title)+2

    data_dict = {'stock_history': stock_history, 'market_history': market_history, 'rf_history': rf_history}

    # Localize datasets
    for name, df in data_dict.items():
        if df.index.tz is not None:
            try:
                df.index = df.index.tz_localize(None)
                print(f'{name} localized: OK.')
            except Exception as e:
                print(f'Error localizing {name}: {e}')
        else:
            print(f'{name} already localized.')
    
    print('=' * (50 + text_length))


    return stock_history, market_history, rf_history

## Function to clean and prepare data and create features
def prepare(stock_history, market_history, rf_history):
    text_title ="PREPARING DATASETS"
    print(f"{'=' * 25} {text_title} {'=' * 25}")
    text_length = len(text_title)+2
    
    
    # Ensure Close column exists before proceeding
    for df, name in zip([stock_history, market_history, rf_history], ['Stock', 'Market', 'Risk-Free']):
        if 'Close' not in df.columns:
            raise ValueError(f"Missing 'Close' column in {name} data")
    
    # Create new return and excess return columns
    stock_history['stock_returns'] = stock_history['Close'].pct_change()
    market_history['market_returns'] = market_history['Close'].pct_change()
    rf_history['annual_rf_rates'] = rf_history['Close'] / 100  # Convert to decimal
    rf_history['daily_rf_rates'] = rf_history['annual_rf_rates'] / 252  # Get daily rates
    
    stock_history['stock_excess'] = stock_history['stock_returns'].sub(rf_history['daily_rf_rates'], fill_value=0)
    market_history['market_excess'] = market_history['market_returns'].sub(rf_history['daily_rf_rates'], fill_value=0)
    
    # Store last values for tomorrow’s returns and create dummy variables
    for df, return_col, tomorrow_col, dummy_col in [
        (stock_history, 'stock_returns', 'stock_tomorrow', 'tomorrow_dummy'),
        (market_history, 'market_returns', 'market_tomorrow', 'tomorrow_dummy')
    ]:
        df[tomorrow_col] = df[return_col].shift(-1)
        df[dummy_col] = df[tomorrow_col].gt(0).astype(int)
    
    for df, rate_col, tomorrow_col in [
        (rf_history, 'annual_rf_rates', 'annual_rf_tomorrow'),
        (rf_history, 'daily_rf_rates', 'daily_rf_tomorrow')
    ]:
        df[tomorrow_col] = df[rate_col].shift(-1)
    
    # Validate column creation
    required_columns = {
        'Stock': ['stock_returns', 'stock_excess', 'stock_tomorrow', 'tomorrow_dummy'],
        'Market': ['market_returns', 'market_excess', 'market_tomorrow', 'tomorrow_dummy'],
        'Risk-Free': ['annual_rf_rates', 'daily_rf_rates', 'annual_rf_tomorrow', 'daily_rf_tomorrow']
    }
    for df, name in zip([stock_history, market_history, rf_history], required_columns.keys()):
        for col in required_columns[name]:
            if col in df.columns and not df[col].isna().all():
                print(f"New column '{col}': OK")
            else:
                print(f"New column '{col}': MISSING")
    
    # Drop unwanted columns and handle missing data
    drop_cols = ['Dividends', 'Stock Splits']
    for df, name in zip([stock_history, market_history, rf_history], ['Stock', 'Market', 'Risk-Free']):
        df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore', inplace=True)
        df.dropna(inplace=True)
        print(f"{name} Data Cleaned: OK")
    
    print('=' * (50 + text_length))

    return stock_history, market_history, rf_history

## Function to filter based on the given period
def date_filter(stock_history, market_history, rf_history):
    text_title ="DATE FILTERING AND CREATING NEW SETS"
    print(f"{'=' * 25} {text_title} {'=' * 25}")
    text_length = len(text_title)+2

    max_period = 5  # Define the period in years
    end_date = pd.Timestamp.now().tz_localize(None)
    start_date = (end_date - pd.DateOffset(years=max_period)).tz_localize(None)
    # Ensure all indices are tz-naive
    for df in [stock_history, market_history, rf_history]:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    data_dict = {'stock_history': stock_history, 'market_history': market_history, 'rf_data': rf_history}

    # Filter datasets
    filtered_stock_history, filtered_market_history, filtered_rf_data = None, None, None
    for name, df in data_dict.items():
        try:
            filtered_df = df.loc[start_date:end_date].dropna()
            print(f'Filtered {name} created: OK')
            if name == 'stock_history':
                filtered_stock_history = filtered_df
            elif name == 'market_history':
                filtered_market_history = filtered_df
            else:
                filtered_rf_data = filtered_df
        except Exception as e:
            print(f'Error filtering {name}: {e}')
    print('=' * (50 + text_length))

    return filtered_stock_history, filtered_market_history, filtered_rf_data

## Function to print df.tail(1), series.iloc[-1]
def check_tail(*args):
    text_title = "CHECK LAST ROW"
    print(f"{'=' * 25} {text_title} {'=' * 25}")
    text_length = len(text_title) + 2
    
    # Get the calling frame
    frame = inspect.currentframe().f_back
    
    # Get the argument names as they were passed to the function
    arg_names = list(frame.f_locals.keys())
    arg_dict = {name: value for name, value in frame.f_locals.items() if name in arg_names}    
    
    for arg in args:
        name = next((name for name, value in arg_dict.items() if value is arg), "Unnamed Input")
        print(f"\nProcessing input: {name}")
        try:
            if isinstance(arg, pd.DataFrame):
                print(f"DataFrame last row:")
                print(arg.tail(1))
            elif isinstance(arg, pd.Series):
                print(f"Series last value:")
                print(arg.iloc[-1])
            else:
                print(f"Unsupported type: {type(arg)}")
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
        
    
    print('=' * (50 + text_length))

## Function to extract returns and rates
def extract_returns(filtered_stock_data, filtered_market_data, filtered_rf_data):
    text_title ="EXTRACT RETURNS"
    print(f"{'=' * 25} {text_title} {'=' * 25}")
    text_length = len(text_title)+2

    # Ensure all dataframes have a datetime index
    for df in [filtered_stock_data, filtered_market_data, filtered_rf_data]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    common_dates = filtered_stock_data.index.intersection(filtered_market_data.index).intersection(filtered_rf_data.index)
    
    if len(common_dates) == 0:
        print("No common dates found across the datasets.")
        return None, None, None, None

    print(f"Found {len(common_dates)} common dates for alignment")
    
    try:
        # Extract returns and rates
        stock_returns = filtered_stock_data.loc[common_dates, 'stock_returns']
        stock_excess =  filtered_stock_data.loc[common_dates, 'stock_excess']
        market_returns = filtered_market_data.loc[common_dates, 'market_returns']
        market_excess = filtered_market_data.loc[common_dates, 'market_excess']

        daily_rf_rates = filtered_rf_data.loc[common_dates, 'daily_rf_rates']
        annual_rf_rates = filtered_rf_data.loc[common_dates, 'annual_rf_rates']
        

        # Verify extractions
        for name, series in [('stock_returns', stock_returns), ('stock_excess', stock_excess), ('market:returns', market_returns), 
                             ('market_excess', market_excess), ('daily_rf_rates', daily_rf_rates), ('annual_rf_rates', annual_rf_rates)]:
            if not series.empty:
                print(f"{name} Extraction: OK (Length: {len(series)})")
            else:
                print(f"{name} Extraction: Failed (Empty series)")

    except KeyError as e:
        print(f"Column not found: {e}")
        return None, None, None, None
    
    except Exception as e:
        print(f"Unexpected error during extraction: {e}")
        return None, None, None, None

    print('=' * (50 + text_length))

    return stock_returns, stock_excess, market_returns, market_excess, daily_rf_rates, annual_rf_rates

## Function to check and print data types

## Function to calculate Beta
def calculate_beta(stock_returns, market_returns):
    text_title = "CALCULATE BETA"
    print(f"{'=' * 25} {text_title} {'=' * 25}")
    text_length = len(text_title) + 2

    # Convert inputs to pandas Series for alignment and handling missing data
    stock_returns = pd.Series(stock_returns)
    market_returns = pd.Series(market_returns)

    # Check for missing or invalid data
    if stock_returns.isnull().any() or market_returns.isnull().any():
        print("Error: Missing values detected in input data.")
        return None

    if len(stock_returns) != len(market_returns):
        print("Error: Mismatched lengths between stock and market returns.")
        return None

    if stock_returns.var() == 0:
        print("Error: Stock returns have zero variance.")
        return None

    if market_returns.var() == 0:
        print("Error: Market returns have zero variance.")
        return None

    # Add constant for intercept in regression
    X = sm.add_constant(market_returns)
    
    # Fit the model
    try:
        model = sm.OLS(stock_returns, X, missing='drop').fit()
        beta = model.params[1]  # Beta is the slope (params[1])
        # Check if beta is valid
        if beta is not None and np.isfinite(beta):
            print("BETA: OK")
            print(f"Calculated Beta: {beta}")
        else:
            print("BETA: Calculation Error")

    except Exception as e:
        print(f"Error during regression calculation: {e}")
        return None

    print('=' * (50 + text_length))

    return beta

## Function to calculate CAPM
def capm_return(beta_stock, current_annual_rf_rate, mean_market_return, current_stock_return):
    text_title ="CALCULATE CAPM"
    print(f"{'=' * 25} {text_title} {'=' * 25}")
    text_length = len(text_title)+2

    # Capm calculation
    capm_return = (current_annual_rf_rate + beta_stock * (mean_market_return - current_annual_rf_rate))    
    
    # Convert to scalar if it's still a Series
    if isinstance(current_stock_return, pd.Series):
        current_stock_return = current_stock_return.item()  # Convert to scalar using .item()

    # Compare CAPM expected return with current stock return
    advice = "UNDERVALUED" if capm_return > current_stock_return else "OVERVALUED"
    
    # Print details (like the old function)
    print(f"\nBeta: {beta_stock}")
    print(f"\ncurrent_annual_rf_rate: {current_annual_rf_rate}")
    print(f"\nmean_market_return: {mean_market_return}")
    print(f"current_market_return: {current_stock_return}\n")
    print(f"Capm: {capm_return}\n")
    print(f"advice: {advice}\n")
    print('=' * (50 + text_length))

    return capm_return, advice
    
## Function to plot CAPM /SML
def sml(current_annual_rf_rate, beta_stock, capm_result, mean_market_return_annualized, current_market_return, mean_stock_return_annualized, current_stock_return, advice):
    text_title ="GRAPH SML"
    print(f"{'=' * 25} {text_title} {'=' * 25}")
    text_length = len(text_title)+2
    # Title: Plot the Security Market Line (SML)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Title: Generate Beta Values and SML Returns
    betas = np.linspace(0, 2, 100)  # Beta values for visualization
    sml_returns = current_annual_rf_rate + betas * (mean_market_return_annualized - current_annual_rf_rate)  # SML formula

    # Title: Plot SML Line
    ax.plot(betas, sml_returns * 100, label="Security Market Line (SML)", color="blue")

    # Title: Scatter Plots for Stock, Risk-Free Rate, and Market Returns
    ax.scatter(beta_stock, capm_result * 100, color="red", s=100,
            label=f"Stock (β={beta_stock:.2f}, Expected Return={capm_result:.2f}%)")

    ax.scatter(0, current_annual_rf_rate * 100, color="green", s=100,
            label=f"Risk-Free Rate ({current_annual_rf_rate:.2f}%)")

    ax.scatter(beta_stock, current_stock_return * 100, color="orange", s=100,
            label=f"Current Stock Return ({current_stock_return:.2f}%)")

    # Title: Indicate Market Positioning
    long_position = advice
    ax.scatter(beta_stock, mean_stock_return_annualized * 100, color="orange", s=100,
            label=f"Mean Stock Return ({mean_market_return_annualized:.2f}%)")


    # Create the annotation first
    text = ax.text(beta_stock, mean_stock_return_annualized * 100, f'Stock is\n{long_position}', fontsize=9, ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.25'))

    # Use adjustText to adjust the text position dynamically to avoid overlap
    adjust_text([text], ax=ax, expand_points=(1.5, 1.5), force_points=0.5, lim=200, only_move={'points': 'xy', 'text': 'xy'})


    # Title: Scatter Plot for Market Returns
    ax.scatter(1, current_market_return * 100 + 0.1, color="purple", s=100,
            label=f"Current Market Return ({current_market_return:.2f}%)")

    ax.scatter(1, mean_market_return_annualized * 100, color="purple", s=100,
            label=f"Mean Market Return ({mean_market_return_annualized:.2f}%)")

    # Title: Configure Plot Labels and Titles
    ax.set_xlabel("Beta (Systematic Risk)", fontsize=12)
    ax.set_ylabel("Expected Return (%)", fontsize=12)
    ax.set_title("CAPM Analysis with Multiple Points", fontsize=14)

    # Title: Display Legend and Grid
    ax.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Title: Display Plot in Jupyter Notebook
    #plt.show()
    st.pyplot(fig)  # Display plot in Streamlit

    print('=' * (50 + text_length))


## Function to plot stock_live data

## Function to calculate Indicators

## Function to analyze trends

## Function to graph indicators

## Function to prepate data for ML models

## Function to split into test and train datasets

## Functions to evaluate models

# ====================DATA AND CALCULATION ==================#

## 1- Load Data 
### 1.1- Historical Data:

#### 1.1.1- Load data
stock_history, market_history, rf_history = load_data(s_ticker, m_ticker, '^FVX', 'max')

## 2- Prepare data and Feature engineering
### 2.1- Localize
stock_history, market_history, rf_history = localize(stock_history, market_history, rf_history)

### 2.2- Prepare data
stock_history, market_history, rf_history = prepare(stock_history, market_history, rf_history)

#### 2.2.1- Check data from prepare function return info and tail values
check_tail(stock_history, market_history, rf_history )

### 2.3- Filter date
filtered_stock_data, filtered_market_data, filtered_rf_data = date_filter(stock_history, market_history, rf_history)

### 2.4- Extract returns and rates
stock_returns, stock_excess, market_returns, market_excess, daily_rf_rates, annual_rf_rates = extract_returns(filtered_stock_data, filtered_market_data, filtered_rf_data)

### 2.4.1- Check last rows from extract function return info and tail values
check_tail(stock_returns, stock_excess, market_returns, market_excess, daily_rf_rates, annual_rf_rates)

## 3- Calculate CAPM
### 3.1- beta
beta_stock = calculate_beta(stock_excess, market_excess)

### 3.2- Annualized market and stock means and current annual rf rate
mean_market_return_annualized = float(market_returns.mean() * 252)  # 252 trading days in a year
mean_stock_return_annualized = float(stock_returns.mean() * 252) # 252 trading days in a year
current_annual_rf_rate = float(annual_rf_rates.iloc[-1])

### 3.3- Current market, stock returns, current_rf_rate
current_market_return= float(market_returns.iloc[-1]) 
current_stock_return = float(stock_returns.iloc[-1])  # Convert to float

### 3.4- CAPM
capm_result, advice = capm_return(beta_stock, current_annual_rf_rate, mean_market_return_annualized, current_stock_return)

## 4- Define time range
time_ranges = {
     "1D": ("1d", "1m"),
     "5D": ("5d", "5m"),
     "1M": ("1mo", "30m"),
     "3M": ("3mo", "1d"),
     "6M": ("6mo", "1d"),
     "YTD": ("ytd", "1d"),
     "1Y": ("1y", "1d"),
     "5Y": ("5y", "1wk"),
     "MAX": ("max", "1mo"),
 }
selected_period = None
# 4- SML graph


### Live Data:

# ======================== DASHBOARD ========================#
## Main layout of page: two columns. Col_1: show main contents, Col_2: Show indicators graphs
### Main title
st.markdown(f"<h1 class='big-font'>📊 Stock Price Predictor <span style='color: blue;'>({s_ticker})</span></h1>", unsafe_allow_html=True)

sml(current_annual_rf_rate, beta_stock, capm_result, mean_market_return_annualized, current_market_return, mean_stock_return_annualized, current_stock_return, advice)
