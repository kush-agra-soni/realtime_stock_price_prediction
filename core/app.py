import plotly.graph_objects as go
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from train import prepare_multivariate_data, build_and_train_model, predict_future


# Initialize session state for predictions
if 'preds' not in st.session_state:
    st.session_state.preds = None
    st.session_state.future_index = None
    st.session_state.last_close = None

# --- load_data FIX ---
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, group_by='ticker', auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data.columns = [col[1] if col[1] else col[0] for col in data.columns]
        except:
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
    def dedupe(cols):
        seen, out = {}, []
        for c in cols:
            seen[c] = seen.get(c, 0) + 1
            out.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
        return out
    data.columns = dedupe(data.columns)
    return data

# --- Streamlit App ---
st.title("📈 Stock Price Predictor")
ticker     = st.text_input("Enter Stock Ticker", "AAPL").upper()
start_date = st.date_input("Start Date", datetime.date(2018, 1, 1))
end_date   = st.date_input("End Date", datetime.date.today())
epochs     = st.sidebar.slider("Number of Epochs", 1, 70, 25)
days       = st.sidebar.slider("Number of Prediction Days", 1, 30, 7)

# ---- Load & Validate Data ----
data = load_data(ticker, start_date, end_date)
if data.empty:
    st.error("No data for given ticker/date.")
    st.stop()
price_cols = ['Open', 'High', 'Low', 'Close']
missing = [c for c in price_cols if c not in data.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()
data[price_cols] = data[price_cols].apply(pd.to_numeric, errors='coerce')
data.dropna(subset=price_cols, inplace=True)

# ---- Prediction Computation ----
if st.button("Predict Prices"):
    features = price_cols + ['Volume']
    n_steps  = 30
    volatility_factor = 0.05  # Adjust this to control volatility

    # Prepare multivariate data
    X, y_open, y_high, y_low, y_close, scaler = prepare_multivariate_data(data[features], n_steps)

    # Print shapes of data for debugging
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y_open: {y_open.shape}, y_high: {y_high.shape}, y_low: {y_low.shape}, y_close: {y_close.shape}")

    # Check if data is None or empty
    if X is None or y_open is None or y_high is None or y_low is None or y_close is None:
        print("Error: One or more target arrays are None.")
        st.stop()

    # Combine all target arrays into a single y array with 4 columns (Open, High, Low, Close)
    y = np.column_stack((y_open, y_high, y_low, y_close))

    # Train the model with proper data
    model = build_and_train_model(X, y, n_steps, len(features), epochs=epochs)

    # Pass volatility_factor to predict_future to add volatility in predictions
    raw_preds = predict_future(model, data, scaler, features, n_steps, days, volatility_factor)

    # Normalize predictions to list of floats (open, high, low, close)
    try:
        arr = raw_preds.tolist()  # Convert predictions into a list of [open, high, low, close]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if not arr:
        st.error("Model returned no valid predictions.")
        st.stop()

    if len(arr) < days:
        st.warning(f"Only {len(arr)} prediction(s); padding last value to reach {days} days.")
        arr += [arr[-1]] * (days - len(arr))
    else:
        arr = arr[:days]

    # Store predictions in session
    st.session_state.last_close = float(data['Close'].iloc[-1])
    last_date = data.index[-1]
    st.session_state.future_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days)
    st.session_state.preds = arr


# ---- Chart Setup ----
today = datetime.date.today()
window_days = 60
chart_start_date = today - datetime.timedelta(days=window_days)
chart_data = data[data.index >= pd.to_datetime(chart_start_date)].copy()

# ---- Plot Combined Chart ----
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart_data.index,
    open=chart_data['Open'],
    high=chart_data['High'],
    low=chart_data['Low'],
    close=chart_data['Close'],
    name="Historical",
    increasing_line_color='green',
    decreasing_line_color='red'
))

if st.session_state.preds:
    preds        = st.session_state.preds
    future_index = st.session_state.future_index
    last_close   = st.session_state.last_close
    pred_open    = [last_close] + [x[0] for x in preds[:-1]]
    pred_close   = [x[3] for x in preds]
    pred_high    = [x[1] for x in preds]
    pred_low     = [x[2] for x in preds]
    fig.add_trace(go.Candlestick(
        x=future_index,
        open=pred_open,
        high=pred_high,
        low=pred_low,
        close=pred_close,
        name="Predicted",
        increasing_line_color='cyan',
        decreasing_line_color='yellow'
    ))


fig.update_layout(
    title=f"{ticker} Price Chart (Last {window_days} Days)" + 
          (f" + Next {days} Days" if st.session_state.preds else ""),
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=600
)
st.plotly_chart(fig, use_container_width=True)