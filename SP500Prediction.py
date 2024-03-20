import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch historical stock data using yfinance.

    Args:
    symbol (str): Ticker symbol of the stock.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: Historical stock data.
    """
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def preprocess_data(stock_data):
    """
    Preprocess historical stock data.

    Args:
    stock_data (pandas.DataFrame): Historical stock data.

    Returns:
    pandas.DataFrame: Preprocessed data.
    """
    # Feature engineering
    stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
    stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=30).std()

    # Drop rows with NaN values
    stock_data.dropna(inplace=True)

    return stock_data

def build_model(X_train, y_train):
    """
    Build a machine learning model.

    Args:
    X_train (pandas.DataFrame): Features for training.
    y_train (pandas.Series): Target variable for training.

    Returns:
    sklearn.ensemble.RandomForestRegressor: Trained Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the machine learning model.

    Args:
    model (sklearn.ensemble.RandomForestRegressor): Trained Random Forest model.
    X_test (pandas.DataFrame): Features for testing.
    y_test (pandas.Series): Target variable for testing.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

def main():
    # Fetch historical data for S&P 500 (^GSPC)
    sp500_data = fetch_stock_data('^GSPC', '2000-01-01', '2022-12-31')

    # Preprocess data
    sp500_data = preprocess_data(sp500_data)

    # Define features and target variable
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility']
    target = 'Adj Close'

    X = sp500_data[features]
    y = sp500_data[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()