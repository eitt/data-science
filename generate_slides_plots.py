import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
import os

# Set style for clean, professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

def save_plot(name):
    plt.tight_layout()
    plt.savefig(f"assets/{name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {name}.png")

# Create assets directory if not exists
if not os.path.exists('assets'):
    os.makedirs('assets')

# 1. Download AMZN data
print("Downloading AMZN data...")
amzn = yf.Ticker("AMZN").history(start="2020-01-01", end="2024-01-01")['Close']

# Plot 1: Raw Price
plt.figure()
amzn.plot(color='#2c3e50', linewidth=2)
plt.title("Amazon (AMZN) Stock Price: 2020 - 2024", fontweight='bold')
plt.ylabel("Price (USD)")
plt.xlabel("Date")
save_plot("amzn_price")

# 2. Decomposition
print("Performing decomposition...")
decomposition = seasonal_decompose(amzn, model='additive', period=252) # Annual period
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))

amzn.plot(ax=axes[0], color='#2c3e50')
axes[0].set_title("Original")

decomposition.trend.plot(ax=axes[1], color='#e67e22')
axes[1].set_title("Trend (Long-term progression)")

decomposition.seasonal.plot(ax=axes[2], color='#27ae60')
axes[2].set_title("Seasonality (Periodic patterns)")

decomposition.resid.plot(ax=axes[3], color='#c0392b', linestyle='none', marker='.')
axes[3].set_title("Residuals (Stochastic Noise)")

save_plot("amzn_decomposition")

# 3. ARIMA Forecast
print("Fitting ARIMA...")
train = amzn.iloc[:-30]
test = amzn.iloc[-30:]
model = pm.auto_arima(train, seasonal=True, m=5, suppress_warnings=True)
forecast, conf_int = model.predict(n_periods=30, return_conf_int=True)

plt.figure()
plt.plot(train.index[-60:], train.iloc[-60:], label="Training Data", color='#2c3e50')
plt.plot(test.index, test.values, label="Actual Price", color='#2980b9', alpha=0.5)
plt.plot(test.index, forecast, label="ARIMA Forecast", color='#e74c3c', linewidth=2)
plt.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], color='#e74c3c', alpha=0.1)
plt.title("Predicting the Future: ARIMA Forecast for AMZN", fontweight='bold')
plt.legend()
save_plot("amzn_arima")

# 4. Portfolio Optimization (AMZN + GOOGL + MSFT)
print("Generating Portfolio Optimization plot...")
tickers = ["AMZN", "GOOGL", "MSFT"]
data = yf.download(tickers, start="2020-01-01", end="2024-01-01")['Close']
returns = np.log(data / data.shift(1)).dropna()

n_ports = 2000
all_weights = np.zeros((n_ports, len(tickers)))
ret_arr = np.zeros(n_ports)
vol_arr = np.zeros(n_ports)
sharpe_arr = np.zeros(n_ports)

for i in range(n_ports):
    weights = np.array(np.random.random(len(tickers)))
    weights = weights / np.sum(weights)
    all_weights[i, :] = weights
    ret_arr[i] = np.sum((returns.mean() * weights) * 252)
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_arr[i] = ret_arr[i] / vol_arr[i]

plt.figure()
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.title('The Efficient Frontier: Balancing Risk and Reward', fontweight='bold')

# Mark Max Sharpe
max_sr_idx = sharpe_arr.argmax()
plt.scatter(vol_arr[max_sr_idx], ret_arr[max_sr_idx], c='red', s=100, edgecolors='black', marker='*', label='Max Sharpe')
plt.legend()
save_plot("portfolio_frontier")

print("All plots generated successfully in assets/ directory.")
