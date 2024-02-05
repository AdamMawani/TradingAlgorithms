import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define portfolio
portfolio = {"AAPL": 0.66, "ENB": 0.66, "QQQ": 0.66, "SPY": 0.66, "NVDA": 0.66, "MSFT": 0.66, "GOOG": 0.66, "MA": 0.66, "VZ": 0.66, "T": 0.66, "AMZN": 0.66, "C": 0.66, "RY": 0.66, "BNS": 0.66, "BAC": 0.66, "UBER": 0.66}

# Fetch historical data for each stock in the portfolio
start_date = "2020-01-01"
end_date = "2023-01-01"

stock_data = {}
for ticker, weight in portfolio.items():
    data = yf.download(ticker, start=start_date, end=end_date)
    stock_data[ticker] = {'data': data, 'weight': weight}

# Calculate daily returns for each stock in the portfolio
returns = pd.DataFrame()
for ticker, stock_info in stock_data.items():
    data = stock_info['data']
    returns[ticker] = data['Adj Close'].pct_change().dropna() * stock_info['weight']

# Drop NaN values
returns = returns.dropna()

# LSTM input preparation
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(returns.values)

# Prepare input data
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, :])
    y.append(scaled_data[i, :])

X, y = np.array(X), np.array(y)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=len(portfolio)))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions
inputs = scaled_data[-60:, :]
inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))
predicted_returns = model.predict(inputs)
predicted_returns = scaler.inverse_transform(predicted_returns)

# Display predicted returns
predicted_returns_df = pd.DataFrame(predicted_returns, columns=returns.columns)
print("Predicted Returns:")
print(predicted_returns_df)

# Note: This example predicts returns, and you may need to adjust it for your specific use case (e.g., predicting prices, optimizing trading strategies).
