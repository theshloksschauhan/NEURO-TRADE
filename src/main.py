from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
from model.data_loader import fetch_data
from model.preprocessing import scale_data, create_sequences
from model.lstm_model import create_model, evaluate_model

def main():
    df = fetch_data("AAPL")

    print(df.tail())

    scaled_data, scaler = scale_data(df[["close"]])
    X, y = create_sequences(scaled_data, window_size=10)

    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=5, batch_size=16, verbose=1)

    evaluate_model(model, X, y)

    model.save("final_lstm_model.keras")
    print("✅ Model saved as final_lstm_model.keras")

if __name__ == "__main__":
    main()
