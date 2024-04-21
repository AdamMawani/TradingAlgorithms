import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

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

    # On-Balance Volume (OBV)
    data['OBV'] = (data['Volume'] * ((data['Close'] - data['Open']) / (data['High'] - data['Low']))).cumsum()

    # Money Flow Index (MFI)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    raw_money_flow = typical_price * data['Volume']
    positive_flow = (raw_money_flow.where(typical_price > typical_price.shift(1), 0)).rolling(window=14).sum()
    negative_flow = (-raw_money_flow.where(typical_price < typical_price.shift(1), 0)).rolling(window=14).sum()
    money_ratio = positive_flow / negative_flow
    data['MFI'] = 100 - (100 / (1 + money_ratio))

    # Moving Average Envelopes
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['Upper_Envelope'] = data['MA_20'] * 1.025
    data['Lower_Envelope'] = data['MA_20'] * 0.975

    # Chaikin Oscillator
    adl = ((2 * data['Close'] - data['High'] - data['Low']) / (data['High'] - data['Low'])) * data['Volume']
    data['ADL'] = adl.cumsum()
    data['EMA_3_ADL'] = data['ADL'].ewm(span=3, adjust=False).mean()
    data['EMA_10_ADL'] = data['ADL'].ewm(span=10, adjust=False).mean()
    data['Chaikin_Oscillator'] = data['EMA_3_ADL'] - data['EMA_10_ADL']

    # Stochastic Oscillator
    data['Lowest_Low'] = data['Low'].rolling(window=14).min()
    data['Highest_High'] = data['High'].rolling(window=14).max()
    data['%K'] = ((data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low'])) * 100
    data['%D'] = data['%K'].rolling(window=3).mean()

    return data

def main():
    ticker = input("Enter the stock ticker: ")
    start_date = input("Enter the start date (yyyy-mm-dd): ")
    end_date = input("Enter the end date (yyyy-mm-dd): ")

    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        print("No data found for the given ticker and date range.")
        return

    data = calculate_indicators(data)

    print("Initial Price:", data.iloc[0]['Close'])
    print("Final Price:", data.iloc[-1]['Close'])
    percent_change = ((data.iloc[-1]['Close'] - data.iloc[0]['Close']) / data.iloc[0]['Close']) * 100
    print("Percent Change:", percent_change, "%")
    print("Volume:", data['Volume'].sum())
    print("Simple Moving Average (SMA):", data['Close'].rolling(window=20).mean().iloc[-1])  # 20-day SMA
    print("Exponential Moving Average (EMA 20):", data['EMA_20'].iloc[-1])  # 20-day EMA
    print("Upper Bollinger Band:", data['Upper_BB'].iloc[-1])
    print("Lower Bollinger Band:", data['Lower_BB'].iloc[-1])
    print("Relative Strength Index (RSI):", data['RSI'].iloc[-1])
    print("MACD:", data['MACD'].iloc[-1])
    print("Signal Line:", data['Signal_Line'].iloc[-1])
    print("Average True Range (ATR):", data['ATR'].iloc[-1])
    print("On-Balance Volume (OBV):", data['OBV'].iloc[-1])
    print("Money Flow Index (MFI):", data['MFI'].iloc[-1])
    print("Upper Moving Average Envelope:", data['Upper_Envelope'].iloc[-1])
    print("Lower Moving Average Envelope:", data['Lower_Envelope'].iloc[-1])

    plot_stock_data(data, ticker)

if __name__ == "__main__":
    main()
