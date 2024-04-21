import yfinance as yf

def get_undervalued_stocks():
    # Define a list to store undervalued stocks
    undervalued_stocks = []

    # Fetch NASDAQ stock tickers
    nasdaq_tickers = yf.Tickers('^IXIC').tickers

    # Iterate over each ticker
    for ticker in nasdaq_tickers:
        try:
            # Get historical market data for the past year
            hist_data = ticker.history(period='1y')
            
            # Calculate the current price
            current_price = hist_data['Close'][-1]
            
            # Calculate the average P/E ratio for the industry
            industry_pe_ratio = calculate_industry_pe_ratio(ticker)

            # Calculate the P/E ratio for the stock
            pe_ratio = calculate_pe_ratio(hist_data, current_price)

            # Check if the stock is undervalued (P/E ratio < industry average)
            if pe_ratio < industry_pe_ratio:
                undervalued_stocks.append((ticker.ticker, pe_ratio))
        except Exception as e:
            print(f"Error processing {ticker.ticker}: {e}")

    return undervalued_stocks

def calculate_pe_ratio(hist_data, current_price):
    # Calculate earnings per share (EPS)
    earnings_per_share = (hist_data['Close'] - hist_data['Dividends']) / hist_data['Shares Outstanding']
    
    # Calculate P/E ratio
    pe_ratio = current_price / earnings_per_share[-1]
    
    return pe_ratio

def calculate_industry_pe_ratio(ticker):
    # You can implement a more sophisticated method to calculate industry P/E ratio
    # For simplicity, we will use a static value here
    industry_pe_ratio = 15.0
    return industry_pe_ratio

# Main function to run the algorithm
def main():
    undervalued_stocks = get_undervalued_stocks()

    # Print the list of undervalued stocks
    if undervalued_stocks:
        print("Undervalued Stocks:")
        for stock, pe_ratio in undervalued_stocks:
            print(f"{stock}: P/E Ratio - {pe_ratio}")
    else:
        print("No undervalued stocks found.")

if __name__ == "__main__":
    main()
