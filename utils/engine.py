import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Utility function to split sequence for training LSTM models
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        X.append(sequence[i:i + n_steps])
        y.append(sequence[i + n_steps])
    return np.array(X), np.array(y)

# Updated: Prepare data for multivariate LSTM model (includes Volume)
def prepare_multivariate_data(df, n_steps):
    vals = df[['Open', 'High', 'Low', 'Close', 'Volume']].values  # Now includes Volume
    scaler = MinMaxScaler()  # Normalize values to range [0, 1]
    scaled = scaler.fit_transform(vals)
    
    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i - n_steps:i])      # n_steps timesteps
        y.append(scaled[i, 3])               # Still predicting 'Close' (index 3)
    
    return np.array(X), np.array(y), scaler

# Optional: Prepare univariate data
def prepare_univariate_data(df, n_steps, feature_col='High'):
    prices = df[feature_col].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    
    X, y = split_sequence(scaled, n_steps)
    return np.array(X), np.array(y), scaler
