import matplotlib
matplotlib.use("TkAgg")  # Force the GUI backend

import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(df: pd.DataFrame, title="Stock Trend Prediction"):
    """
    Plot the stock price and model predictions as buy/sell markers.
    """
    # Safety check: is the DataFrame empty or missing required columns?
    if df.empty or 'Close' not in df.columns or 'Predicted' not in df.columns:
        print("❌ DataFrame is invalid or missing required columns.")
        return

    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='black')

    # Predicted uptrend → green arrow
    up_days = df[df['Predicted'] == 1]
    plt.scatter(up_days['Date'], up_days['Close'], marker='^', color='green', label='Predicted Up', alpha=0.8)

    # Predicted downtrend → red arrow
    down_days = df[df['Predicted'] == 0]
    plt.scatter(down_days['Date'], down_days['Close'], marker='v', color='red', label='Predicted Down', alpha=0.8)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    print("✅ Plot is ready. Displaying now...")
    plt.show()

    input("🟢 Press Enter to exit after viewing the plot.")
