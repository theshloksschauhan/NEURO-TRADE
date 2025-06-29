import pandas as pd
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to stock price DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']

    Returns:
        pd.DataFrame: DataFrame with new indicator columns
    """
    # Add all indicators automatically
    df = ta.add_all_ta_features(
        df,
        open="Open", high="High", low="Low", close="Close", volume="Volume",
        fillna=True
    )

    return df
