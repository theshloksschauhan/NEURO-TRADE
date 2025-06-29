import os
import pandas as pd
import requests
from io import StringIO

def fetch_data(ticker: str) -> pd.DataFrame:
    """
    Fetch stock data from Alpha Vantage (free TIME_SERIES_DAILY endpoint).
    """
    print(f"Fetching data for {ticker} from Alpha Vantage (free endpoint)...")

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("❌ ALPHAVANTAGE_API_KEY environment variable not set.")

    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}&outputsize=compact&apikey={api_key}&datatype=csv"
    )

    response = requests.get(url)
    if response.status_code != 200:
        raise ConnectionError(f"❌ Failed to fetch data: {response.status_code}")

    if response.text.strip().startswith("{"):
        print("----- API CSV Response (First 5 lines) -----")
        print(response.text[:300])
        print("-------------------------------------------")
        raise ValueError("❌ Alpha Vantage returned JSON. This likely means your API key has hit rate limits or endpoint requires premium.")

    df = pd.read_csv(StringIO(response.text))

    if 'timestamp' not in df.columns:
        raise ValueError("❌ 'timestamp' column not found in API response. Check API key or rate limit.")

    df.rename(columns={"timestamp": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(f"data/raw/{ticker}_alpha.csv", index=False)
    print(f"✅ Saved data to data/raw/{ticker}_alpha.csv")

    return df
