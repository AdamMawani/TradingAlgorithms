import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def prepare_data(stock_data, window_size=10):
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1,1))

    # Create sequences
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape data for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def build_ffnn_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

def predict(model, X_test):
    return model.predict(X_test)

if __name__ == "__main__":
    # User input for stock ticker
    ticker = input("Enter a stock ticker symbol (e.g., 'AAPL' for Apple Inc.): ")

    # Fetch historical data
    stock_data = get_stock_data(ticker, '2020-01-01', '2024-01-01')

    # Prepare data
    X, y, scaler = prepare_data(stock_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train feedforward neural network model
    ffnn_model = build_ffnn_model((X_train.shape[1],))
    train_model(ffnn_model, X_train, y_train)

    # Build and train LSTM model
    lstm_model = build_lstm_model((X_train.shape[1], 1))
    train_model(lstm_model, X_train, y_train)

    # Predict stock prices using both models
    ffnn_predictions = predict(ffnn_model, X_test)
    lstm_predictions = predict(lstm_model, X_test)

    # Inverse scaling for predictions
    ffnn_predictions = scaler.inverse_transform(ffnn_predictions.reshape(-1, 1)).flatten()
    lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Print example predictions
    print("Feedforward Neural Network Predictions:")
    for i in range(5):
        print("Predicted:", ffnn_predictions[i], "Actual:", y_test[i])

    print("\nLSTM Predictions:")
    for i in range(5):
        print("Predicted:", lstm_predictions[i], "Actual:", y_test[i])
