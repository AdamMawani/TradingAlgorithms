# Stock Portfolio and S&P 500 Performance Simulation

This Python program utilizes Monte Carlo simulation with geometric Brownian motion to forecast the future performance of a user-defined stock portfolio and the S&P 500 index. The simulation calculates the daily close price percentage for each stock in the portfolio and the S&P 500, and then performs a Monte Carlo simulation to generate multiple possible future scenarios.

## Prerequisites

Before running the program, make sure to have the required Python libraries installed. You can install them using the following:

```bash
pip install yfinance numpy matplotlib pandas
```

## How to Use It
1. Import the necessary libraries:

```
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

2. Define the functions for fetching stock data, performing Monte Carlo simulation, and plotting the results.

~~~
# Function to get stock data and calculate daily close price percentage
def get_stock_data(ticker, start_date, end_date):
    #Implementation...

#Function for Monte Carlo simulation with geometric Brownian motion
def monte_carlo_simulation_gbm(data, num_simulations, forecast_period, days=252):
    #Implementation...

#Function to plot the average of Monte Carlo simulations
def plot_monte_carlo_average(data, simulations, label):
    #Implementation...
~~~

3. Define your stock portfolio and set the start and end dates for the simulation.

```
#Define portfolio as a dictionary with tickers and corresponding percentages
portfolio = {"AAPL": 0.4, "TSLA": 0.4, "NVDA": 0.2}

#Loop through each stock in the portfolio
daily_percentage_dict_portfolio = None
start_date_portfolio = "2022-01-01"
end_date_portfolio = "2023-10-01"
```

4. Fetch and process stock data for each stock in the portfolio.

```
for stock_ticker, weight in portfolio.items():
    #Implementation...

#Calculate the weighted average close prices across all stocks in the portfolio
daily_percentage_dict_portfolio['Average'] = daily_percentage_dict_portfolio.sum(axis=1)
```

5. Perform Monte Carlo simulation for the portfolio.

```
#Monte Carlo simulation with geometric Brownian motion for the portfolio
num_simulations_portfolio = 1000
forecast_period_portfolio = 66  #Approximately 3 months (assuming 22 business days per month)
portfolio_simulations = monte_carlo_simulation_gbm(daily_percentage_dict_portfolio['Average'], num_simulations_portfolio, forecast_period_portfolio)
```

6. Plot the results for the portfolio.

```
#Plot the average of Monte Carlo simulations for the portfolio
plot_monte_carlo_average(daily_percentage_dict_portfolio['Average'], portfolio_simulations, label="Portfolio")
```

7. Perform Monte Carlo simulation for the S&P 500.

```
#Get data for ^GSPC
sp500_data = yf.download('^GSPC', start=start_date_portfolio, end=end_date_portfolio)
sp500_close_prices_percentage = (sp500_data['Close'] / sp500_data['Open'].iloc[0]) * 100

#Monte Carlo simulation with geometric Brownian motion for ^GSPC
num_simulations_sp500 = 1000
forecast_period_sp500 = 66  #Approximately 3 months (assuming 22 business days per month)
sp500_simulations = monte_carlo_simulation_gbm(sp500_close_prices_percentage, num_simulations_sp500, forecast_period_sp500)
```

8. Plot the results for the S&P 500.

```
#Plot the average of Monte Carlo simulations for ^GSPC
plot_monte_carlo_average(sp500_close_prices_percentage, sp500_simulations, label="S&P 500")
```

9. Add labels and legend, then display the plot.

```
#Add labels and legend
plt.xlabel("Date")
plt.ylabel("Close Price Percentage")
plt.title("Stock Portfolio and S&P 500 Performance with Monte Carlo Simulation")
plt.legend()
plt.show()
```

Feel free to customize the stock portfolio, simulation parameters, and plotting options according to your needs.
