import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Portfolio can be adjusted based on selections and weightings
portfolio = {
    "ENB": 0.15, "QQQ": 0.15, "SPY": 0.05, "NVDA": 0.1, "MSFT": 0.07,
    "GOOG": 0.05, "VZ": 0.07, "T": 0.05, "AMZN": 0.07, "C": 0.03,
    "RY": 0.05, "BNS": 0.03, "BAC": 0.03, "MA": 0.05, "UBER": 0.05
}

start_date = "2014-01-01"
end_date = "2024-01-01"

# Data Retrieval
stock_data = {ticker: yf.Ticker(ticker).history(start=start_date, end=end_date) for ticker in portfolio}

# Data Validation
missing_tickers = set(portfolio.keys()) - set(stock_data.keys())
if missing_tickers:
    raise ValueError(f"Missing data for tickers: {missing_tickers}")

# Calculate Returns
returns = pd.DataFrame({ticker: data['Close'].pct_change() * portfolio[ticker] 
                        for ticker, data in stock_data.items()})
returns['Portfolio'] = returns.sum(axis=1)
returns = returns.dropna()

# ARIMA Modeling
model = auto_arima(returns['Portfolio'], start_p=1, start_q=1,
                   test='adf', max_p=3, max_q=3, m=1,
                   d=None, seasonal=False, start_P=0, 
                   D=0, trace=True, error_action='ignore',  
                   suppress_warnings=True, stepwise=True)

print(model.summary())

# Out-of-sample Forecasting
n_periods = 30
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(returns.index, returns['Portfolio'], label='Actual Returns')
plt.plot(returns.index, model.fittedvalues(), color='red', label='ARIMA Fitted Values')

# Plot future predictions
future_dates = pd.date_range(start=returns.index[-1], periods=n_periods+1, freq='D')[1:]
plt.plot(future_dates, fc, color='green', label='ARIMA Forecast')
plt.fill_between(future_dates, confint[:, 0], confint[:, 1], color='green', alpha=0.1)

plt.legend()
plt.title('ARIMA Model for Portfolio Returns')
plt.show()

# Performance Evaluation
mse = mean_squared_error(returns['Portfolio'], model.fittedvalues())
rmse = np.sqrt(mse)
mae = mean_absolute_error(returns['Portfolio'], model.fittedvalues())

print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")

# Risk Analysis
annual_returns = returns['Portfolio'].mean() * 252
annual_volatility = returns['Portfolio'].std() * np.sqrt(252)
sharpe_ratio = annual_returns / annual_volatility

print(f"Annual Returns: {annual_returns:.2%}")
print(f"Annual Volatility: {annual_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")