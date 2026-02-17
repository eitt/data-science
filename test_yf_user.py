import yfinance as yf
import pandas as pd
import numpy as np

start_date = "2018/01/01"
end_date = "2026/02/15"

try:
    start_date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    print(f"Downloading AAPL from {start_date_str} to {end_date_str}...")
    df = yf.download("AAPL", start=start_date_str, end=end_date_str, progress=False)
    print("Success!")
    print(df.tail())
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
