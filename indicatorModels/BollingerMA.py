import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
from statsmodels.graphics.tsaplots import plot_acf

# Define the stock ticker and the period for which we want to download data
ticker = 'AAPL'  # Apple Inc.
start_date = '2020-01-01'
end_date = '2023-01-01'

# Download stock data
data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows of the dataset
print(data.head())

# Data Preprocessing
# Calculate the daily returns
data['Daily Return'] = data['Adj Close'].pct_change()

# Calculate the moving averages
data['50 Day MA'] = data['Adj Close'].rolling(window=50).mean()
data['200 Day MA'] = data['Adj Close'].rolling(window=200).mean()

# Calculate Bollinger Bands
data['20 Day MA'] = data['Adj Close'].rolling(window=20).mean()
data['20 Day Std'] = data['Adj Close'].rolling(window=20).std()
data['Upper Band'] = data['20 Day MA'] + (2 * data['20 Day Std'])
data['Lower Band'] = data['20 Day MA'] - (2 * data['20 Day Std'])

# Calculate RSI
delta = data['Adj Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Calculate Exponential Moving Average (EMA)
data['12 Day EMA'] = data['Adj Close'].ewm(span=12, adjust=False).mean()
data['26 Day EMA'] = data['Adj Close'].ewm(span=26, adjust=False).mean()

# Calculate rolling volatility (standard deviation of daily returns)
data['Rolling Volatility'] = data['Daily Return'].rolling(window=21).std()

# Drop the NaN values created by the rolling window operations
data.dropna(inplace=True)

# Time Series Analysis
# Plot Adjusted Close price, Moving Averages, and Bollinger Bands
plt.figure(figsize=(14, 7))
plt.plot(data['Adj Close'], label='Adjusted Close Price', linewidth=1.5)
plt.plot(data['50 Day MA'], label='50 Day MA', linestyle='--', linewidth=1.0)
plt.plot(data['200 Day MA'], label='200 Day MA', linestyle='--', linewidth=1.0)
plt.plot(data['Upper Band'], label='Upper Bollinger Band', linestyle='-.', linewidth=1.0)
plt.plot(data['Lower Band'], label='Lower Bollinger Band', linestyle='-.', linewidth=1.0)
plt.fill_between(data.index, data['Lower Band'], data['Upper Band'], color='grey', alpha=0.3)
plt.title('Adjusted Close Price, Moving Averages, and Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Plot RSI
plt.figure(figsize=(10, 6))
data['RSI'].plot()
plt.axhline(30, linestyle='--', alpha=0.5, color='red')
plt.axhline(70, linestyle='--', alpha=0.5, color='red')
plt.title('Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.grid()
plt.show()

# Distribution of Daily Returns with Skewness and Kurtosis
plt.figure(figsize=(10, 6))
sns.histplot(data['Daily Return'], kde=True, bins=50)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid()
plt.show()
print(f"Skewness: {skew(data['Daily Return'])}")
print(f"Kurtosis: {kurtosis(data['Daily Return'])}")

# Rolling Volatility
plt.figure(figsize=(10, 6))
data['Rolling Volatility'].plot()
plt.title('Rolling Volatility (21-Day)')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.grid()
plt.show()

# Autocorrelation of Daily Returns
plot_acf(data['Daily Return'].dropna(), lags=30)
plt.title('Autocorrelation of Daily Returns')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid()
plt.show()

# Jointplot of Adjusted Close Price vs Volume
sns.jointplot(x='Volume', y='Adj Close', data=data, kind='scatter', alpha=0.5)
plt.suptitle('Adjusted Close Price vs Volume', y=1.02)
plt.show()

# Pairplot of key variables
sns.pairplot(data[['Adj Close', 'Volume', 'Daily Return', '50 Day MA', '200 Day MA', 'RSI', 'Rolling Volatility']])
plt.suptitle('Pairplot of Key Variables', y=1.02)
plt.show()

# Save the processed data to a CSV file
data.to_csv('processed_stock_data_extended.csv')