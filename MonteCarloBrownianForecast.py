import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to get stock data and calculate daily close price percentage
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    close_prices = stock_data['Close']
    start_price = close_prices.iloc[0]
    close_prices_percentage = (close_prices / start_price) * 100
    return close_prices_percentage

# Function for Monte Carlo simulation with geometric Brownian motion
def monte_carlo_simulation_gbm(data, num_simulations, forecast_period, days=252):
    returns = data.pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()

    simulations = np.zeros((forecast_period, num_simulations))
    simulations[0, :] = data.iloc[-1]

    for day in range(1, forecast_period):
        drift = (mu - 0.5 * sigma**2) * (days / 252)
        diffusion = sigma * np.sqrt(days / 252) * np.random.normal(size=num_simulations)
        simulations[day, :] = simulations[day - 1, :] * np.exp(drift + diffusion)

    return simulations

# Function to plot the average of Monte Carlo simulations
def plot_monte_carlo_average(data, simulations, label):
    plt.plot(data.index, data, label=label + " (Actual)", linewidth=2, color='blue')
    
    # Calculate and plot the average of simulations
    average_simulation = np.mean(simulations, axis=1)
    plt.plot(data.index[-1] + pd.to_timedelta(np.arange(1, len(average_simulation) + 1), unit='D'), average_simulation, label=label + " (Average Prediction)", linestyle='--', linewidth=2, color='red')

# Define portfolio as a dictionary with tickers and corresponding percentages
portfolio = {"AAPL": 0.66, "ENB": 0.66, "QQQ": 0.66, "SPY": 0.66, "NVDA": 0.66, "MSFT": 0.66, "GOOG": 0.66, "MA": 0.66, "VZ": 0.66, "T": 0.66, "AMZN": 0.66, "C": 0.66, "RY": 0.66, "BNS": 0.66, "BAC": 0.66, "UBER": 0.66}

# Loop through each stock in the portfolio
daily_percentage_dict_portfolio = None
start_date_portfolio = "2022-01-01"
end_date_portfolio = "2023-10-01"

for stock_ticker, weight in portfolio.items():
    close_prices_percentage = get_stock_data(stock_ticker, start_date_portfolio, end_date_portfolio)

    if daily_percentage_dict_portfolio is None:
        daily_percentage_dict_portfolio = pd.DataFrame(close_prices_percentage * weight)
    else:
        daily_percentage_dict_portfolio[stock_ticker] = close_prices_percentage * weight

# Calculate the weighted average close prices across all stocks in the portfolio
daily_percentage_dict_portfolio['Average'] = daily_percentage_dict_portfolio.sum(axis=1)

# Monte Carlo simulation with geometric Brownian motion for the portfolio
num_simulations_portfolio = 1000
forecast_period_portfolio = 66  # Approximately 3 months (assuming 22 business days per month)
portfolio_simulations = monte_carlo_simulation_gbm(daily_percentage_dict_portfolio['Average'], num_simulations_portfolio, forecast_period_portfolio)

# Plot the average of Monte Carlo simulations for the portfolio
plot_monte_carlo_average(daily_percentage_dict_portfolio['Average'], portfolio_simulations, label="Portfolio")

# Get data for ^GSPC
sp500_data = yf.download('^GSPC', start=start_date_portfolio, end=end_date_portfolio)
sp500_close_prices_percentage = (sp500_data['Close'] / sp500_data['Open'].iloc[0]) * 100

# Monte Carlo simulation with geometric Brownian motion for ^GSPC
num_simulations_sp500 = 1000
forecast_period_sp500 = 66  # Approximately 3 months (assuming 22 business days per month)
sp500_simulations = monte_carlo_simulation_gbm(sp500_close_prices_percentage, num_simulations_sp500, forecast_period_sp500)

# Plot the average of Monte Carlo simulations for ^GSPC
plot_monte_carlo_average(sp500_close_prices_percentage, sp500_simulations, label="S&P 500")

# Add labels and legend
plt.xlabel("Date")
plt.ylabel("Close Price Percentage")
plt.title("Stock Portfolio and S&P 500 Performance with Monte Carlo Simulation")
plt.legend()
plt.show()
