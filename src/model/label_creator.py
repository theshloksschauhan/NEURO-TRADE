import pandas as pd

def create_trend_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a binary target label: 1 if price goes up next day, else 0.
    """
    df = df.copy()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df
