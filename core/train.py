import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === Constants ===
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

# === Data Preparation ===
def prepare_multivariate_data(df, n_steps, features=FEATURE_COLS):
    vals = df[features].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vals)

    X, y_open, y_high, y_low, y_close = [], [], [], [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i - n_steps:i])
        y_open.append(scaled[i, 0])  # Predict 'Open'
        y_high.append(scaled[i, 1])  # Predict 'High'
        y_low.append(scaled[i, 2])   # Predict 'Low'
        y_close.append(scaled[i, 3]) # Predict 'Close'
    return np.array(X), np.array(y_open), np.array(y_high), np.array(y_low), np.array(y_close), scaler

# === Model Building ===
# === Model Building ===
def build_and_train_model(X, y, n_steps, n_features, epochs=10, batch_size=32, lr=0.001):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4))  # Output 4 values: Open, High, Low, Close

    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    return model



def predict_future(model, data, scaler, features, n_steps, n_future_days, volatility_factor=0.05):
    last_seq = data[features].tail(n_steps).values
    seq_scaled = scaler.transform(last_seq)
    preds_scaled_open, preds_scaled_high, preds_scaled_low, preds_scaled_close = [], [], [], []

    for _ in range(n_future_days):
        next_open, next_high, next_low, next_close = model.predict(seq_scaled[np.newaxis, ...], verbose=0)[0]

        # Add volatility by introducing noise
        noise_open = np.random.normal(0, volatility_factor * next_open)  # Random noise for Open
        noise_high = np.random.normal(0, volatility_factor * next_high)  # Random noise for High
        noise_low = np.random.normal(0, volatility_factor * next_low)    # Random noise for Low
        noise_close = np.random.normal(0, volatility_factor * next_close)  # Random noise for Close

        # Apply noise
        next_open += noise_open
        next_high += noise_high
        next_low += noise_low
        next_close += noise_close

        preds_scaled_open.append(next_open)
        preds_scaled_high.append(next_high)
        preds_scaled_low.append(next_low)
        preds_scaled_close.append(next_close)

        # Update sequence for next prediction
        new_step = seq_scaled[-1].copy()
        new_step[0] = next_open  # Replace 'Open'
        new_step[1] = next_high  # Replace 'High'
        new_step[2] = next_low   # Replace 'Low'
        new_step[3] = next_close # Replace 'Close'
        seq_scaled = np.vstack([seq_scaled[1:], new_step])

    # Inverse transform all predictions
    full_preds_scaled = np.zeros((n_future_days, len(features)))
    full_preds_scaled[:, 0] = preds_scaled_open
    full_preds_scaled[:, 1] = preds_scaled_high
    full_preds_scaled[:, 2] = preds_scaled_low
    full_preds_scaled[:, 3] = preds_scaled_close
    preds = scaler.inverse_transform(full_preds_scaled)
    return preds


