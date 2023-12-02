import yfinance as yf
import numpy as np
import pandas as pd

def calculate_portfolio_var(portfolio, start_date, end_date):
    data = yf.download(list(portfolio.keys()), start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()

    weights = np.array(list(portfolio.values()))
    weighted_returns = returns.dot(weights)

    portfolio_var = np.var(weighted_returns)
    return portfolio_var

def calculate_var(portfolio, confidence_level, start_date, end_date):
    z_score = -1.96  # For 95% confidence level, assuming normal distribution

    portfolio_var = calculate_portfolio_var(portfolio, start_date, end_date)
    var = np.sqrt(portfolio_var) * z_score

    return var

if __name__ == "__main__":
    # Example portfolio
    portfolio = {"AAPL": 0.1, "TSLA": 0.9}

    # Historical data start and end dates
    start_date = "2022-01-01"
    end_date = "2023-01-01"

    # Confidence level for VaR calculation (e.g., 95%)
    confidence_level = 0.95

    # Calculate VaR
    var = calculate_var(portfolio, confidence_level, start_date, end_date)

    print(f"Value at Risk (VaR) at {confidence_level * 100}% confidence level: {var:.2%}")
