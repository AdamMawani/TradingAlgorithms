import yfinance as yf

def get_stocks():
    undervalued_stocks = []
    nasdaq_tickers = yf.Tickers('^IXIC').tickers

    for ticker in nasdaq_tickers:
        try:
            hist_data = ticker.history(period='1y')
            current_price = hist_data['Close'][-1]
            industry_pe_ratio = calculate_industry_pe_ratio(ticker)
            pe_ratio = calculate_pe_ratio(hist_data, current_price)
            if pe_ratio < industry_pe_ratio:
                undervalued_stocks.append((ticker.ticker, pe_ratio))
        except Exception as e:
            print(f"Error processing {ticker.ticker}: {e}")

    return undervalued_stocks

def calculate_pe_ratio(hist_data, current_price):
    earnings_per_share = (hist_data['Close'] - hist_data['Dividends']) / hist_data['Shares Outstanding']
    pe_ratio = current_price / earnings_per_share[-1]
    return pe_ratio

def calculate_industry_pe_ratio(ticker):
    industry_pe_ratio = 15.0
    return industry_pe_ratio

def main():
    undervalued_stocks = get_stocks()
    if undervalued_stocks:
        print("Undervalued Stocks:")
        for stock, pe_ratio in undervalued_stocks:
            print(f"{stock}: P/E Ratio - {pe_ratio}")
    else:
        print("No undervalued stocks found.")

if __name__ == "__main__":
    main()