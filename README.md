# Understanding Supervised Learning: An Interactive Streamlit App

This is an interactive Streamlit web application designed to teach the fundamental concepts of supervised machine learning and practical time-series & finance analyses. It provides hands-on, visual demonstrations of optimization, regression, classification, neural networks, stochastic processes (heartbeat), and portfolio analysis using real market data.

## New Features (recent updates)

- Heartbeat simulation: a Gaussian-pulse model with tunable BPM, amplitude, pulse width (σ), baseline (β₀), optional slow amplitude modulation ("in-love"), and noise. The UI displays the expected model coefficients in color.
- Finance tab improvements: calculates simple & log returns, annualized mean & volatility, correlation heatmap, and a Monte Carlo portfolio simulation (random portfolios with max-Sharpe and min-vol representatives). Also includes stationarity checks and optional ARIMA forecasting (requires `pmdarima`).

## About the Author

Developed by **Dr. Leonardo H. Talero-Sarmiento** (Ph.D., Universidad Autónoma de Bucaramanga). His work covers mathematical modeling, data analytics, operations research, manufacturing systems, and applied machine learning.

## Learning Modules (high level)

- Introduction
- Gradient Descent
- Manual Linear Fit (cost surface)
- Train vs Test (Overfitting)
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)
- Neural Network Training (Backpropagation)
- Time Series (Data Leaks)
- TS Analysis (rolling windows)
- Finance (Stochastic Processes & Portfolio Analysis)

## Finance analyses included

- Price downloads from Yahoo Finance via `yfinance` (robust to missing adjusted close column).
- Simple and log returns computation.
- Annualized mean return and volatility (approx, using 252 trading days).
- Correlation heatmap of log-returns.
- Monte Carlo random-portfolio simulation (weights drawn from Dirichlet), scatter of return vs volatility, representative portfolios (max Sharpe and min volatility), and cumulative portfolio price plots.

## How to Run

1.  Prerequisites:

    - Python 3.10+ recommended
    - `pip`

2.  Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Run the app:

    ```bash
    streamlit run app.py
    ```

5.  Open the provided URL in your browser (usually `http://localhost:8501`).

## Notes & Tips

- If `yfinance` downloads fail for some symbols, the app will attempt to use the `Close` column or another numeric column. Rate limits from Yahoo Finance may still cause intermittent failures.
- ARIMA forecasting requires `pmdarima` — install it if you plan to use the auto-ARIMA feature.
- The `requirements.txt` file pins `altair==4.2.2` to avoid compatibility issues with Streamlit's Altair usage.

## Project Structure

```
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
```

## License & Credits

This project is for educational use. Please credit the author when using material from this repository.
