import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def fetch_stock_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data[ticker] = stock.history(start=start_date, end=end_date)
    return data

def plot_all_stocks_on_single_graph(data):
    plt.figure(figsize=(10, 8))
    for ticker, stock_data in data.items():
        plt.plot(stock_data.index, stock_data['Close'], label=ticker)
    plt.title('Stock Prices for All Tickers')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_stock_data(data, ticker):
    plt.figure(figsize=(10, 8))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, data['EMA_20'], label='EMA (20)', color='red')
    plt.plot(data.index, data['Upper_BB'], label='Upper Bollinger Band', color='green', linestyle='--')
    plt.plot(data.index, data['Lower_BB'], label='Lower Bollinger Band', color='green', linestyle='--')
    plt.title(f'{ticker} Stock Price and Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_indicators(data):
    # Exponential Moving Average (EMA)
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['std_20'] = data['Close'].rolling(window=20).std()
    data['Upper_BB'] = data['MA_20'] + (data['std_20'] * 2)
    data['Lower_BB'] = data['MA_20'] - (data['std_20'] * 2)

    # Relative Strength Index (RSI)
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))

    # Moving Average Convergence Divergence (MACD)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Average True Range (ATR)
    data['TR'] = pd.DataFrame(data = (data['High'] - data['Low']).abs(), index = data.index)
    data['ATR'] = data['TR'].rolling(window=14).mean()

    return data

def main():
    tickers = input("Enter the stock tickers separated by comma: ").strip().split(',')
    start_date = input("Enter the start date (yyyy-mm-dd): ")
    end_date = input("Enter the end date (yyyy-mm-dd): ")

    data = fetch_stock_data(tickers, start_date, end_date)
    if not data:
        print("No data found for the given tickers and date range.")
        return

    for ticker, stock_data in data.items():
        data[ticker] = calculate_indicators(stock_data)

    # Plotting all stocks on a single graph
    plot_all_stocks_on_single_graph(data)

    # Plotting each stock on a separate graph
    for ticker, stock_data in data.items():
        plot_stock_data(stock_data, ticker)

if __name__ == "__main__":
    main()