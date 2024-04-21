import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler

def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def build_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(50, input_shape=input_shape, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Main function
def main():
    symbol = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2024-01-01'
    time_steps = 10

    stock_data = fetch_stock_data(symbol, start_date, end_date)
    close_prices = stock_data['Close'].values

    scaled_data, scaler = preprocess_data(close_prices)
    X, y = prepare_data(scaled_data, time_steps)
    input_shape = (X.shape[1], 1)

    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_rnn_model(input_shape)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    loss = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)

if __name__ == "__main__":
    main()
