import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_content(url):
    """
    Scrapes the content from the given URL.

    Args:
        url (str): The URL of the financial report.

    Returns:
        str: The extracted text content or None if an error occurred.
    """
    try:
        logging.info(f"Scraping content from URL: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        
        return text
    except requests.RequestException as e:
        logging.error(f"Request error while scraping content: {e}")
    except Exception as e:
        logging.error(f"Error scraping content: {e}")
    
    return None

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text.

    Args:
        text (str): The text to analyze.

    Returns:
        float: The sentiment rating on a scale of 0-10.
    """
    logging.info("Analyzing sentiment of the content.")
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']
    
    sentiment_rating = (compound_score + 1) * 5  # Normalize to 0-10 scale
    
    logging.info(f"Sentiment score (compound): {compound_score}, rating: {sentiment_rating}")
    return sentiment_rating

def main(url):
    """
    Main function to scrape content and analyze sentiment.

    Args:
        url (str): The URL of the financial report.
    """
    content = scrape_content(url)
    
    if content:
        sentiment_rating = analyze_sentiment(content)
        print(f"Sentiment Rating: {sentiment_rating}")
    else:
        print("Failed to retrieve content from the URL")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sentiment_analysis.py <URL>")
    else:
        url = sys.argv[1]
        main(url)