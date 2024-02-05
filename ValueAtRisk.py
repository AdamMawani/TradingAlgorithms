import yfinance as yf
import numpy as np
import pandas as pd
from tabulate import tabulate

def calculate_portfolio_var(portfolio, start_date, end_date):
    data = yf.download(list(portfolio.keys()), start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    weights = np.array(list(portfolio.values()))
    weighted_returns = returns.dot(weights)
    portfolio_var = np.var(weighted_returns)
    return portfolio_var

def calculate_var(portfolio, start_date, end_date):
    z_score = -1.96  # For 95% confidence level, assuming normal distribution
    portfolio_var = calculate_portfolio_var(portfolio, start_date, end_date)
    var = np.sqrt(portfolio_var) * z_score
    return var

def calculate_stock_var(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    stock_var = np.var(returns)
    var = np.sqrt(stock_var) * -1.96
    return var

if __name__ == "__main__":
    # Example portfolio
    portfolio = {"AAPL": 0.66, "ENB": 0.66, "QQQ": 0.66, "SPY": 0.66, "NVDA": 0.66, "MSFT": 0.66, "GOOG": 0.66, "MA": 0.66, "VZ": 0.66, "T": 0.66, "AMZN": 0.66, "C": 0.66, "RY": 0.66, "BNS": 0.66, "BAC": 0.66, "UBER": 0.66}
    stocks = ["AAPL", "ENB", "QQQ", "SPY", "NVDA", "MSFT", "GOOG", "MA", "VZ", "T", "AMZN", "C", "RY", "BNS", "BAC", "UBER"]

    # Historical data start and end dates
    start_date = "2022-01-01"
    end_date = "2023-01-01"

    # Confidence level for VaR calculation (e.g., 95%)
    confidence_level = 0.95
    printable_confidence = 100 * confidence_level

    # Calculate VaR
    var = calculate_var(portfolio, start_date, end_date)

    # Create a table for the portfolio VaR
    table_data = [["Portfolio", f"{printable_confidence}%", f"{var:.2%}"]]
    
    # Calculate and display VaR for each stock
    for stock in stocks:
        stock_var = calculate_stock_var(stock, start_date, end_date)
        table_data.append([stock, f"{printable_confidence}%", f"{stock_var:.2%}"])

    # Print the table
    headers = ["Asset", "Confidence Level", "Value at Risk"]
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
