import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def scrape_content(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from HTML
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        
        return text
    except Exception as e:
        print("Error scraping content:", e)
        return None

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']
    
    # Normalize the compound score to a scale of 0-10
    sentiment_rating = (compound_score + 1) * 5
    
    return sentiment_rating

def main():
    # Example URL of the financial report
    url = "example.com/financial-report"
    
    # Scrape content from the URL
    content = scrape_content(url)
    
    if content:
        # Analyze sentiment of the scraped content
        sentiment_rating = analyze_sentiment(content)
        
        print("Sentiment Rating:", sentiment_rating)
    else:
        print("Failed to retrieve content from the URL")

if __name__ == "__main__":
    main()