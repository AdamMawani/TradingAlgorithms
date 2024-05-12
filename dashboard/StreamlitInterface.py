import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def fetch_stock_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data[ticker] = stock.history(start=start_date, end=end_date)
    return data

def plot_all_stocks_on_single_graph(data):
    plt.figure(figsize=(10, 6))
    for ticker, stock_data in data.items():
        plt.plot(stock_data.index, stock_data['Close'], label=ticker)
    plt.title('Stock Prices for All Tickers')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot()

def plot_stock_data(data, ticker, indicators):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    
    for indicator in indicators:
        if indicator == 'EMA':
            plt.plot(data.index, data['EMA_20'], label='EMA (20)', color='red')
        elif indicator == 'BB':
            plt.plot(data.index, data['Upper_BB'], label='Upper Bollinger Band', color='green', linestyle='--')
            plt.plot(data.index, data['Lower_BB'], label='Lower Bollinger Band', color='green', linestyle='--')
        elif indicator == 'RSI':
            plt.plot(data.index, data['RSI'], label='RSI', color='purple')
        elif indicator == 'MACD':
            plt.plot(data.index, data['MACD'], label='MACD', color='orange')
            plt.plot(data.index, data['Signal_Line'], label='Signal Line', color='pink')
        elif indicator == 'ATR':
            plt.plot(data.index, data['ATR'], label='ATR', color='brown')

    plt.title(f'{ticker} Stock Price and Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot()

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
    st.title("Stock Analysis Interface")

    tickers = st.text_input("Enter the stock tickers separated by comma (e.g., AAPL, MSFT):").strip().split(',')
    start_date = st.date_input("Enter the start date:", datetime(2020, 1, 1))
    end_date = st.date_input("Enter the end date:", datetime.today())

    if st.button("Fetch Data"):
        data = fetch_stock_data(tickers, start_date, end_date)
        if not data:
            st.error("No data found for the given tickers and date range.")
            return

        for ticker, stock_data in data.items():
            data[ticker] = calculate_indicators(stock_data)

        # Plotting all stocks on a single graph
        plot_all_stocks_on_single_graph(data)

        # Comparative Analysis
        selected_tickers = st.multiselect("Select tickers for comparative analysis:", tickers)
        if selected_tickers:
            for ticker in selected_tickers:
                selected_indicators = st.multiselect(f"Select indicators to plot for {ticker}:", ['EMA', 'BB', 'RSI', 'MACD', 'ATR'])
                plot_stock_data(data[ticker], ticker, selected_indicators)

        # Additional Statistics
        st.subheader("Additional Statistics:")
        for ticker, stock_data in data.items():
            st.write(f"**{ticker}**")
            st.write(f"Average Daily Return: {stock_data['Close'].pct_change().mean()}")
            st.write(f"Cumulative Return: {stock_data['Close'][-1] / stock_data['Close'][0] - 1}")
            st.write(f"Volatility (Standard Deviation of Daily Returns): {stock_data['Close'].pct_change().std()}")

if __name__ == "__main__":
    main()