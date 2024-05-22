import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to fetch data
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    return data['Adj Close']

# Function to calculate moving averages
def moving_average(data, window):
    return data.rolling(window=window).mean()

# Function to calculate relative strength index (RSI)
def rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to calculate MACD
def macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# Function to plot data
def plot_data(ticker, data, ma_windows):
    plt.figure(figsize=(14, 10))

    # Plotting adjusted close price
    plt.subplot(3, 1, 1)
    plt.plot(data, label=f'{ticker} Price', color='blue')
    for window in ma_windows:
        ma = moving_average(data, window)
        plt.plot(ma, label=f'{window}-day MA')
    plt.title(f'{ticker} Adjusted Close Price with Moving Averages')
    plt.legend()

    # Plotting RSI
    plt.subplot(3, 1, 2)
    rsi_data = rsi(data)
    plt.plot(rsi_data, label='RSI', color='red')
    plt.axhline(70, color='black', linestyle='--', linewidth=1)
    plt.axhline(30, color='black', linestyle='--', linewidth=1)
    plt.title(f'{ticker} Relative Strength Index (RSI)')
    plt.legend()

    # Plotting MACD
    plt.subplot(3, 1, 3)
    macd_line, signal_line = macd(data)
    plt.plot(macd_line, label='MACD Line', color='green')
    plt.plot(signal_line, label='Signal Line', color='orange')
    plt.title(f'{ticker} MACD')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main function to perform analysis
def market_analysis(tickers, start, end):
    data = fetch_data(tickers, start, end)
    
    if isinstance(data, pd.Series):
        # Single ticker
        ticker = tickers if isinstance(tickers, str) else tickers[0]
        print(f'Analyzing {ticker}')
        plot_data(ticker, data, ma_windows=[20, 50, 200])
    else:
        # Multiple tickers
        for ticker in data.columns:
            print(f'Analyzing {ticker}')
            plot_data(ticker, data[ticker], ma_windows=[20, 50, 200])

# Define parameters
tickers = ['AAPL', 'MSFT', 'GOOGL', '^GSPC']
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# Perform market analysis
market_analysis(tickers, start_date, end_date)