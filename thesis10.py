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
import statsmodels.api as sm
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
# ======================== FUNCTIONS ========================

## Function to Fetch Live Stock Data
@st.cache_data
def load_live(symbol):
    stock = yf.Ticker(symbol)
    df_d = stock.history(period="1d", interval="1m")  # Fetch today's data with 1-minute intervals
    df_d.reset_index(inplace=True)
    return df_d

## Function to get historical stock data
def load_history(symbol, period, interval):
    stock = yf.Ticker(symbol)
    df_hm =stock.history(period=period, interval=interval)
    df_hm.reset_index(inplace=True)
    return df_hm

## Function to get risk free rate
# def rf_rate(symbol):
#     rate = yf.Ticker(symbol)
#     df_rate = rate.history(period="1d", interval="1d")
#     df_rate.reset_index(inplace=True)
#     print(symbol)

#     return df_rate

## Function to calculate CAPM
def calculate_capm(stock_data, market_data, rf_data):
    # Calculate returns
    stock_returns = stock_data['Close'].pct_change().dropna()
    market_returns = market_data['Close'].pct_change().dropna()
    annual_rf_rate = rf_data['Close']/100 # change to decimal
    daily_rf_rate = annual_rf_rate / 252
    
    # Ensure the columns are aligned
    aligned_returns = pd.concat([stock_returns, market_returns, annual_rf_rate, daily_rf_rate], axis=1).dropna()
    stock_returns = aligned_returns.iloc[:, 0]
    market_returns = aligned_returns.iloc[:, 1]
    annual_rf_rate = aligned_returns.iloc[:, 2]
    daily_rf_rate = aligned_returns.iloc[:, 3]

       
    # Calculate excess returns
    excess_stock_returns = stock_returns - daily_rf_rate
    excess_market_returns = market_returns -daily_rf_rate

     # Calculate annualized returns and risk-free rate
    excess_market_return_annual = ((1 + excess_market_returns).prod()) ** (252/len(excess_market_returns)) - 1
    
    # # Calculate beta
    # covariance = np.cov(stock_returns, market_returns)[0][1]
    # market_variance = np.var(market_returns)
    # beta = covariance / market_variance
    # risk_free_rate = risk_free_rate/100

    # Estimate beta using OLS regression
    X = sm.add_constant(excess_market_returns)  # Add intercept
    model = sm.OLS(excess_stock_returns, X, missing='drop').fit()
    beta = model.params[0]

    # Calculate expected return using CAPM
    actual_market_return = market_returns.iloc[-1]
    market_mean = market_returns.mean()
    expected_return = risk_free_rate + beta * ( market_mean- risk_free_rate)
    actual_return = stock_returns.iloc[-1]


    return beta, market_mean, actual_market_return, expected_return, actual_return

## Function to plot CAPM


def sml(risk_free_rate, market_mean, actual_market_return, beta, expected_return, actual_return):
    # Generate beta values (0 to 2.0 for visualization)
    betas = np.linspace(0, 2, 100)
    
    # Calculate SML returns using CAPM formula
    sml_returns = risk_free_rate + betas * (market_mean - risk_free_rate)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))   
     
    ax.plot(betas, sml_returns, label='Security Market Line (SML)', color='blue')
    
    # Plot market portfolio (beta=1)
    ax.scatter(1, actual_market_return, color='red', s=100, label='Market Portfolio (β=1)')
    
    # Plot risk-free rate (beta=0)
    ax.scatter(0, risk_free_rate, color='green', s=100, label=f'Risk-Free Rate ({risk_free_rate:.1f}%)')
    
    # Plot stock's position
    position = "UNDERVALUED" if actual_return > expected_return else "OVERVALUED"
    ax.scatter(beta, actual_return, color='orange', s=100, 
               label=f'Stock (β={beta:.2f}, Return={actual_return:.2f}%)')
    
    # Annotations with shorter arrow
    # Add text box
    ax.text(beta, expected_return, f'Stock is {position}\n(SML Return: {expected_return*100:.2f}%)\n(Actual Return: {actual_return*100:.2f}%)', fontsize=9, ha='right', va='bottom',
        bbox=dict(facecolor='white', edgecolor='black', alpha=actual_return+0.2))
    
    
    # Formatting
    ax.set_xlabel('Beta (Systematic Risk)', fontsize=12)
    ax.set_ylabel('Expected Return (%)', fontsize=12)
    ax.set_title('Security Market Line (SML)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis to percentage format
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.2f}%'))
    
    # Adjust layout
    plt.tight_layout()
    
    st.pyplot(fig)



# ======================== LOAD DATA ========================

# Get risk free rate
rf_symbol = risk_free_options[rf_ticker]
data_rf =  rf_rate(rf_symbol)
# Fetch Live and Historical Stock Data
data_ls = load_live(s_ticker) #Stock Live Data
data_li = load_live(i_ticker) # Index Live Data
data_hs = load_history(s_ticker) # Stock historical data
data_hi = load_history(i_ticker) # Index historical data
# Convert Date column to datetime format
data_ls["Datetime"] = pd.to_datetime(data_ls["Datetime"]) #Live stock
data_li["Datetime"] = pd.to_datetime(data_li["Datetime"]) #Live index
data_hi["Datetime"] = pd.to_datetime(data_hi["Date"]) # historical stock
data_hs["Datetime"] = pd.to_datetime(data_hs["Date"]) # historical stock

# ================ POST-DATA LOAD FUNCTIONS =================

# Function to calculate Indicators
def indicators (data):
    """
    Calculates Simple Moving Averages (SMA), Relative Strength Index (RSI),
    and Moving Average Convergence Divergence (MACD) for the given stock data.

    Args:
        data (pd.DataFrame): DataFrame containing stock data with a 'Close' column.

    Returns:
        pd.DataFrame: The input DataFrame with added columns for SMA_50, SMA_200, RSI, MACD, and Signal.
                      Returns None if the input data is empty.
    """
    if data.empty:
        st.error("No data available for the selected stock and date range.")
        return None  # Return None when data is empty
    else:
        # Add MACD and RSI calculations
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        # Calculate RSI
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Calculate MACD
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Data preparation for ML models
        # Data preparation for ML models
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal']
        X = data[features].dropna()
        y = X['Close']
        X = X.drop('Close', axis=1)
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
        
        return data, scaler_X, scaler_y, X_scaled, y_scaled, X_train, X_test, y_train, y_test
# Models :
## Function to test and train into train and test
def train (X_train, y_train):
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train.ravel())

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)

    return rf_model, lr_model, lstm_model

## Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name, is_lstm=False):
    if is_lstm:
        X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        y_pred_scaled = model.predict(X_test_reshaped).flatten()
    else:
        y_pred_scaled = model.predict(X_test).flatten()
    
    # Inverse transform predictions
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    if model_name == "LSTM":
        logloss = log_loss(y_test, y_pred)  # Predictions are probabilities
        y_pred_binary = np.round(y_pred)  # Convert probabilities to binary
    else:
        # For other models, we can convert predictions to 0 or 1 based on a threshold (e.g., 0.5)
        logloss = log_loss(y_test, np.clip(y_pred, 1e-15, 1 - 1e-15))  # Clip values for numerical stability
        y_pred_binary = np.round(y_pred)

    cm = confusion_matrix(y_test, y_pred_binary)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred_binary)
    return {
        'Model': model_name,
        'MAE': f"{mae:.2f}",
        'MSE': f"{mse:.2f}",
        'RMSE': f"{rmse:.2f}",
        'R² Score': f"{r2:.4f}",
        'Accuracy': f"{accuracy:.4f}",
        'Log Loss': f"{logloss:.4f}",
        'Confusion Matrix': f"{cm:.4f}",
    }
    
# ======================== DASHBOARD ========================

# Dynamic visual for Price (Stock, Index) and CAPM Display
ticker = yf.Ticker(s_ticker)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if not data_ls.empty: # Live stock price
        latest_pls = data_ls["Close"].iloc[-1] #lastes price live
        st.metric(label=f"Current Price of {s_ticker}", value=f"${latest_pls:.2f}")
    else:
        st.error("No data available for the entered stock symbol.")
with col2:
    if not data_hs.empty: # predicted price
        latest_phs= ticker.fast_info['last_price']#data_hs["Close"].iloc[-1] #lastes price live
        st.metric(label=f"Average Predicted Price {s_ticker}", value=f"${latest_phs:.2f}")
    else:
        st.error("No data available for the entered Index symbol.")
with col3:
    if not data_li.empty:
        latest_pli = data_li["Close"].iloc[-1]
        st.metric(label=f"Current Price of {i_ticker}", value=f"${latest_pli:.2f}")
    else:
        st.error("No data available for the entered Index symbol.")
with col4:
    if not data_rf.empty:
        latest_rf =data_rf["Close"].iloc[-1]
        st.metric(label=f"Risk-Free rate{rf_symbol}", value=f"%{latest_rf:.2f}")
    else:
        st.error("No data available for the Risk Free rate.")
with col5:
    if not data_hs.empty and not data_hi.empty and not data_rf.empty:
        risk_free_rate = data_rf['Close'].iloc[-1] / 100  # Convert to decimal
        beta, market_mean, actual_market_return, expected_return, actual_return = calculate_capm(data_hs, data_hi, risk_free_rate)
        
        st.metric(label=f"CAPM ({s_ticker})", value=f"%{expected_return*100:.2f}")
    else:
        st.error("Insufficient data to calculate CAPM.")

sml(risk_free_rate, market_mean, actual_market_return, beta, expected_return, actual_return)

# Tabs for Model Evaluation and Prediction

tab1, tab2 = st.tabs(["📊 Model Evaluation", "📈 Prediction"])
## ================== TAB 1: MODEL EVALUATION ===============
# calculate and add indicators columns
stock_history_indicator, scaler_X, scaler_y, X_scaled, y_scaled, X_train, X_test, y_train, y_test = indicators (data_hs)

# train models
rf_model, lr_model, lstm_model = train(X_train, y_train)
with tab1:
    st.subheader(" 📊Model Evaluation Report")

    # Create separate scalers for features and target
    x_scaler = scaler_X
    y_scaler = scaler_y
    X_scaled = X_scaled
    y_scaled = y_scaled

    # Generate evaluation results
    results = [
        evaluate_model(rf_model, X_test, y_test, "Random Forest"),
        evaluate_model(lr_model, X_test, y_test, "Linear Regression"),
        evaluate_model(lstm_model, X_test, y_test, "LSTM", is_lstm=True)
    ]
    
    # Display metrics table
    st.subheader("Performance Metrics Comparison")
    results_df = pd.DataFrame(results).set_index('Model')
    st.dataframe(results_df, use_container_width=True)

    # ============== VISUALIZATION SECTION ==============
    st.header("🔍 Prediction Analysis")
    
    def create_prediction_plots(model, X_test, y_test, model_name, is_lstm=False):
        if is_lstm:
            X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            y_pred_scaled = model.predict(X_test_reshaped).flatten()
        else:
            y_pred_scaled = model.predict(X_test).flatten()
        
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        residuals = y_actual - y_pred

        # Create subplots
        fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=[
                            f"Actual vs Predicted ({model_name})",
                            f"Residual Distribution ({model_name})",
                            f"Prediction Error ({model_name})"
                        ])
        
        # Actual vs Predicted plot
        fig.add_trace(go.Scatter(
            x=y_actual, y=y_pred, 
            mode='markers', 
            name='Predictions',
            marker=dict(color='#636EFA', opacity=0.7)
        ), row=1, col=1)
        
        # Add perfect prediction line
        fig.add_trace(go.Scatter(
            x=[min(y_actual), max(y_actual)],
            y=[min(y_actual), max(y_actual)],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ), row=1, col=1)

        # Residual histogram
        fig.add_trace(go.Histogram(
            x=residuals, 
            nbinsx=50,
            marker_color='#00CC96',
            name='Residuals'
        ), row=1, col=2)
        
        # Residual vs Predicted plot
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals, 
            mode='markers',
            name='Residuals',
            marker=dict(color='#EF553B', opacity=0.5)
        ), row=1, col=3)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=3)

        fig.update_layout(
            height=400, 
            showlegend=False,
            margin=dict(l=40, r=40, b=40, t=40)
        )
        return fig

    # Create tabs for each model's visualizations
    tab3, tab4, tab5 = st.tabs(["Random Forest", "Linear Regression", "LSTM"])
    
    with tab3:
        st.plotly_chart(create_prediction_plots(rf_model, X_test, y_test, "Random Forest"), 
                    use_container_width=True, key="rf_plot")
    
    with tab4:
        st.plotly_chart(create_prediction_plots(lr_model, X_test, y_test, "Linear Regression"), 
                    use_container_width=True, key="lr_plot")
    
    with tab5:
        st.plotly_chart(create_prediction_plots(lstm_model, X_test, y_test, "LSTM", is_lstm=True), 
                    use_container_width=True, key="lstm_plot")

    # Add this section after your evaluation metrics display
    st.header("📝 AI-Powered Performance Interpretation")

    def generate_interpretation(results_df):
        # Extract the best model based on the lowest RMSE (converted to float for comparison)
        best_model = results_df.loc[results_df['RMSE'].str.replace('$', '').astype(float).idxmin()]
        
        interpretation = []
        
        # General comparison
        interpretation.append(f"**🏆 Top Performer**: {best_model.name} with lowest RMSE ({best_model['RMSE']})")
        interpretation.append(f"**📉 Closest Predictions**: {results_df['MAE'].idxmin()} (Average error: {results_df['MAE'].min()})")
        interpretation.append(f"**📈 Best Fit**: {results_df['R² Score'].idxmax()} (R²: {results_df['R² Score'].max()})")
        
        # Model-specific insights
        for model in results_df.index:
            metrics = results_df.loc[model]
            insights = [
                f"**{model} Analysis**:",
                f"- Explains {float(metrics['R² Score'])*100:.2f}% of price variance (R²)",
                f"- Predictions typically within ±{metrics['MAE']} of actual prices",
                f"- Larger errors up to ±${float(metrics['RMSE'].replace('$',''))*2:.2f} possible"
            ]
            
            if model == "Linear Regression":
                insights.append("- Linear relationships dominate this dataset")
                insights.append("- Simplicity enables reliable baseline predictions")
                
            if model == "Random Forest":
                insights.append("- Handles non-linear patterns effectively")
                insights.append("- May capture complex market interactions")
                
            if model == "LSTM":
                insights.append("- Temporal patterns present but less impactful")
                insights.append("- Consider increasing sequence length for better temporal learning")
                
            interpretation.extend(insights)
        
        # Recommendations
        interpretation.append("\n**🔍 Recommendations**:")
        interpretation.append("- For trading strategies: Use Linear Regression for its consistency")
        interpretation.append("- For risk analysis: Consider Random Forest's error distribution")
        interpretation.append("- For long-term forecasting: Explore LSTM with more temporal data")
        
        return "\n\n".join(interpretation), best_model

    # Generate interpretation and extract best model
    interpretation_text, best_model = generate_interpretation(results_df)

    # Create expandable interpretation section
    with st.expander("**See AI Analysis of Model Performance**", expanded=True):
        st.markdown(interpretation_text)
        
        # Add dynamic visual guidance
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Overall Model", value=best_model.name)
            st.caption("Selected by lowest RMSE score")
            
        with col2:
            st.metric("Most Consistent", 
                    value=results_df['MAE'].idxmin(),
                    delta=f"MAE: {results_df['MAE'].min()}")
## ================TAB 2: STOCK DATA & PREDICTION============
with tab2:
### Indicator selection
    col6, col7, col8, col9 = st.columns(4)

with col6:
    show_volume = st.checkbox("Show Volume", value=True)
with col7:
    show_sma_20 = st.checkbox("Show SMA 20", value=True)
with col8:
    show_sma_50 = st.checkbox("Show SMA 50", value=True)
with col9:
    show_rsi = st.checkbox("Show RSI", value=False)

# # 📌 **Moving Averages (SMA 50 & 200)**
# if len(data) >= 200:
#     data["SMA_50"] = data["Close"].rolling(window=50).mean()
#     data["SMA_200"] = data["Close"].rolling(window=200).mean()

# # 📌 **Candlestick Chart**
# st.subheader("📈 Live Candlestick Chart")
# fig_candle = go.Figure()
# fig_candle.add_trace(go.Candlestick(x=data["Datetime"], 
#                                     open=data["Open"], 
#                                     high=data["High"], 
#                                     low=data["Low"], 
#                                     close=data["Close"], 
#                                     name="Candlestick"))
# if "SMA_50" in data and "SMA_200" in data:
#     fig_candle.add_trace(go.Scatter(x=data["Datetime"], y=data["SMA_50"], mode='lines', name="SMA 50", line=dict(color="orange")))
#     fig_candle.add_trace(go.Scatter(x=data["Datetime"], y=data["SMA_200"], mode='lines', name="SMA 200", line=dict(color="purple")))
# fig_candle.update_layout(title=f"{ticker} Live Candlestick Chart", xaxis_title="Time", yaxis_title="Price", xaxis_rangeslider_visible=False)
# st.plotly_chart(fig_candle)

# # 📌 **MACD Calculation**
# if len(data) >= 26:
#     data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
#     data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
#     data["MACD"] = data["EMA_12"] - data["EMA_26"]
#     data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

#     st.subheader("📊 MACD Indicator")
#     fig_macd = go.Figure()
#     fig_macd.add_trace(go.Scatter(x=data["Datetime"], y=data["MACD"], mode='lines', name="MACD", line=dict(color="blue")))
#     fig_macd.add_trace(go.Scatter(x=data["Datetime"], y=data["Signal"], mode='lines', name="Signal", line=dict(color="red")))
#     fig_macd.update_layout(title="MACD Indicator", xaxis_title="Time", yaxis_title="MACD Value")
#     st.plotly_chart(fig_macd)

# # 📌 **RSI Calculation**
# if len(data) >= 14:
#     delta = data["Close"].diff(1)
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)

#     avg_gain = gain.rolling(window=14).mean()
#     avg_loss = loss.rolling(window=14).mean()

#     rs = avg_gain / avg_loss
#     data["RSI"] = 100 - (100 / (1 + rs))

#     st.subheader("📊 RSI Indicator")
#     fig_rsi = go.Figure()
#     fig_rsi.add_trace(go.Scatter(x=data["Datetime"], y=data["RSI"], mode="lines", name="RSI", line=dict(color="green")))
#     fig_rsi.update_layout(title="RSI Indicator", xaxis_title="Time", yaxis_title="RSI Value",
#                           shapes=[{"type": "line", "y0": 70, "y1": 70, "x0": min(data["Datetime"]), "x1": max(data["Datetime"]),
#                                    "line": {"color": "red", "dash": "dash"}},
#                                   {"type": "line", "y0": 30, "y1": 30, "x0": min(data["Datetime"]), "x1": max(data["Datetime"]),
#                                    "line": {"color": "blue", "dash": "dash"}}])
#     st.plotly_chart(fig_rsi)

# # 📌 **Display Live Data Table**
# st.subheader("📜 Live Stock Data")
# st.dataframe(data[["Datetime", "Close"]].dropna())

# # 📌 **Auto-Refresh Live Data**
# refresh_interval = st.slider("Auto Refresh Interval (Seconds)", 5, 60, 5)
# st.write(f"Auto-refreshing every {refresh_interval} seconds...")
# time.sleep(refresh_interval)
# st.experimental_rerun()
