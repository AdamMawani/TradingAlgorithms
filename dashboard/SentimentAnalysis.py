import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

def get_news_titles(stock):
    news_titles = []
    try:
        url = f"https://www.bloomberg.com/search?query={stock}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Replace 'css selector' with the actual CSS selector for the news titles
            news_headlines = soup.select('.news-story__headline')
            for headline in news_headlines:
                news_titles.append(headline.text.strip())
    except Exception as e:
        print(f"Error fetching news titles: {e}")
    return news_titles

def analyze_sentiment(news_titles):
    """
    Function to analyze sentiment of news articles using NLTK Vader.
    """
    sentiments = []
    for title in news_titles:
        sentiment_score = sid.polarity_scores(title)['compound']
        sentiments.append(sentiment_score)
    return sentiments

def calculate_average_sentiment(sentiments):
    """
    Function to calculate the average sentiment score.
    """
    if sentiments:
        average_sentiment = sum(sentiments) / len(sentiments)
        return average_sentiment
    else:
        return 0

def get_sentiment_label(score):
    """
    Function to determine sentiment label based on the sentiment score.
    """
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def main():
    stock = input("Enter the stock symbol to analyze sentiment: ")
    news_titles = get_news_titles(stock)
    if news_titles:
        sentiments = analyze_sentiment(news_titles)
        average_sentiment = calculate_average_sentiment(sentiments)
        sentiment_label = get_sentiment_label(average_sentiment)
        print(f"Average sentiment for {stock}: {average_sentiment:.2f} ({sentiment_label})")
    else:
        print("No news articles found for the given stock.")

if __name__ == "__main__":
    main()