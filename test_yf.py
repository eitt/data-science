import yfinance as yf
import pandas as pd
try:
    print("Testing yf.download('AAPL')...")
    df = yf.download('AAPL', start='2023-01-01', end='2023-01-10', progress=False)
    print("Success!")
    print(df.head())
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
