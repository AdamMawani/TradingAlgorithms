import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf

sns.set(style="whitegrid")

indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P 500, Dow Jones, Nasdaq, Russell 2000

def fetch_data(tickers, period='1y'):
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, period=period)
    return data

def calculate_daily_change(data):
    for ticker, df in data.items():
        df['Daily_Change'] = df['Close'].pct_change() * 100
        df.dropna(inplace=True)
    return data

def calculate_average_change(data):
    changes = pd.DataFrame()
    for ticker, df in data.items():
        changes[ticker] = df['Daily_Change']
    changes['Average_Change'] = changes.mean(axis=1)
    return changes

def plot_average_change(changes):
    plt.figure(figsize=(14, 7))
    plt.title('Average Daily Change of Major Indices')
    plt.xlabel('Date')
    plt.ylabel('Average Daily Change (%)')
    plt.plot(changes.index, changes['Average_Change'], color='blue', label='Average Daily Change')
    plt.legend()
    plt.show()

# Plot candlestick chart for the first index as an example
def plot_candlestick(data, ticker):
    df = data[ticker]
    df['Date'] = df.index
    mpf.plot(df, type='candle', style='charles', title=f'{ticker} Candlestick Chart',
             ylabel='Price', volume=True)

# Main execution
if __name__ == "__main__":
    data = fetch_data(indices)
    data = calculate_daily_change(data)
    average_changes = calculate_average_change(data)
    
    # Plot average daily change
    plot_average_change(average_changes)
    
    # Plot candlestick chart for the first index as an example
    plot_candlestick(data, indices[0])