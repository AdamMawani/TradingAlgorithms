import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

# Define portfolio
portfolio = {"ENB": 0.15, "QQQ": 0.15, "SPY": 0.05, "NVDA": 0.1, "MSFT": 0.07, "GOOG": 0.05, "VZ": 0.07, "T": 0.05, "AMZN": 0.07, "C": 0.03, "RY": 0.05, "BNS": 0.03, "BAC": 0.03, "MA": 0.05, "UBER": 0.05}

# Fetch historical data for each stock in the portfolio
start_date = "2022-01-01"
end_date = "2023-01-01"

stock_data = {}
for ticker, weight in portfolio.items():
    data = yf.download(ticker, start=start_date, end=end_date)
    stock_data[ticker] = {'data': data, 'weight': weight}

# Calculate daily returns for each stock in the portfolio
returns = pd.DataFrame()
for ticker, stock_info in stock_data.items():
    data = stock_info['data']
    weight = stock_info['weight']
    returns[ticker] = data['Adj Close'].pct_change().dropna() * 10 * weight

# Calculate the portfolio returns as the sum of individual stock returns
returns['Portfolio'] = returns.sum(axis=1)

# Fit GARCH(1,1) model to the portfolio returns
model = arch_model(returns['Portfolio'], vol='Garch', p=1, q=1)
results = model.fit()

# Display model summary
print(results.summary())

# Plot portfolio returns and conditional volatility
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(returns.index, returns['Portfolio'], label='Portfolio Returns')
ax.plot(returns.index, results.conditional_volatility, label='Conditional Volatility (GARCH(1,1))', color='red')

ax.set(title='Portfolio Returns and GARCH(1,1) Conditional Volatility',
       xlabel='Date', ylabel='Returns / Volatility')
ax.legend()

plt.show()
