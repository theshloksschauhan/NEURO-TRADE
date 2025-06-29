import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(df: pd.DataFrame, feature_cols, target_col='Target', window_size=10):
    """
    Prepares the data in sequences for LSTM input.
    """
    df = df.dropna()
    data = df[feature_cols].values
    target = df[target_col].values

    # Normalize the feature data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i])
        y.append(target[i])

    X, y = np.array(X), np.array(y)
    return X, y, scaler
