import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(df, window_size=60, target_col='Close'):
    df = df.sort_values('Date')
    data = df[[target_col]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # reshape for LSTM input

    return X, y, scaler
