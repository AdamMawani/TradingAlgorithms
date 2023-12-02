import yfinance as yf
import numpy as np
from scipy.stats import norm

def black_scholes(stock_price, strike_price, time_to_maturity, volatility, interest_rate, option_type='call'):
    """
    Calculate the Black-Scholes option price.

    Parameters:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_maturity (float): Time to maturity in years.
        volatility (float): Stock price volatility (annualized).
        interest_rate (float): Risk-free interest rate (annualized).
        option_type (str): Type of option, either 'call' or 'put'.

    Returns:
        float: Option price.
    """
    d1 = (np.log(stock_price / strike_price) + (interest_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)

    if option_type == 'call':
        option_price = stock_price * norm.cdf(d1) - strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price

# Example usage:
ticker = "AAPL"  # Replace with your desired stock ticker
start_date = "2022-01-01"
end_date = "2023-01-01"

# Fetch historical data
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Calculate stock volatility (annualized)
stock_volatility = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1)).std() * np.sqrt(252)

# Set other parameters
stock_price = stock_data['Adj Close'][-1]
strike_price = 150  # Replace with your desired strike price
time_to_maturity = 30 / 252  # 30 days to maturity, converted to years
interest_rate = 0.02  # 2% annualized interest rate

# Calculate call option price
call_option_price = black_scholes(stock_price, strike_price, time_to_maturity, stock_volatility, interest_rate, option_type='call')
print(f"Call Option Price: {call_option_price}")

# Calculate put option price
put_option_price = black_scholes(stock_price, strike_price, time_to_maturity, stock_volatility, interest_rate, option_type='put')
print(f"Put Option Price: {put_option_price}")
