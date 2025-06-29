import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def evaluate_model(model, X_test, y_test):
    """
    Evaluate LSTM model and print metrics.
    """
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    print(f"📊 Evaluation Metrics:\n  RMSE: {rmse:.4f}\n  MAE: {mae:.4f}")
    return predictions, rmse, mae

def plot_predictions(y_test, predictions, title="Prediction vs Actual"):
    """
    Plot model predictions vs actual.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_model(model, path="models/lstm_model.h5"):
    """
    Save the model in HDF5 format.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"✅ Model saved to: {path}")
