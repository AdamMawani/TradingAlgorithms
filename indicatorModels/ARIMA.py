import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Portfolio can be adjusted based on selections and weightings
portfolio = {"ENB": 0.15, "QQQ": 0.15, "SPY": 0.05, "NVDA": 0.1, "MSFT": 0.07, "GOOG": 0.05, "VZ": 0.07, "T": 0.05, "AMZN": 0.07, "C": 0.03, "RY": 0.05, "BNS": 0.03, "BAC": 0.03, "MA": 0.05, "UBER": 0.05}

start_date = "2020-01-01"
end_date = "2023-01-01"

stock_data = {}
for ticker, weight in portfolio.items():
    data = yf.download(ticker, start=start_date, end=end_date)
    stock_data[ticker] = {'data': data, 'weight': weight}

returns = pd.DataFrame()
for ticker, stock_info in stock_data.items():
    data = stock_info['data']
    returns[ticker] = data['Adj Close'].pct_change().dropna() * stock_info['weight']

returns = returns.dropna()
returns['Portfolio'] = returns.sum(axis=1)

order = (1, 1, 1)
model = ARIMA(returns['Portfolio'], order=order)
results = model.fit()

print(results.summary())

plt.figure(figsize=(12, 6))
plt.plot(returns['Portfolio'], label='Actual Returns')
plt.plot(results.fittedvalues, color='red', label='ARIMA Predicted Returns')
plt.legend()
plt.title('ARIMA Model for Portfolio Returns')
plt.show()