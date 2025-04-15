import os
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import feedparser
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re


# 25 pts
def fetch_news_for_one_stock(ticker_symbol, driver, within_days):
    """
    Fetch recent news headlines for a single stock from Yahoo Finance using Selenium.

    Parameters:
        ticker_symbol (str): The stock ticker (e.g., 'AAPL').
        driver (webdriver.Chrome): An instance of a Selenium WebDriver used to load pages.
        within_days (int): Only keep news articles published within this number of days.

    Returns:
        list of dict: Each dict contains 'ticker', 'title', 'link', 'date', and 'content'.
    
    Instructions:
    1. Construct the RSS feed URL using the ticker symbol.
    2. Parse the feed to extract entries and filter articles based on the published date.
    3. For each article, use Selenium to load the link and wait for the JavaScript content to render.
    4. Use BeautifulSoup to extract the main content from the loaded page.
    """
    news_list = []
    cutoff_date = datetime.now() - timedelta(days=within_days)

    # RSS feed
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_symbol}&region=US&lang=en-US"
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        print(f"[WARNING] No entries found in RSS feed for {ticker_symbol}")
        return []

    print(f"[INFO] Processing news for {ticker_symbol} (last {within_days} days)")

    for entry in feed.entries:
        # TODO: Filter articles based on the published date.
        published = datetime(*entry.published_parsed[:6])
        if published < cutoff_date:
            continue

        title = entry.title
        link = entry.link

        try:
            # Set timeout and load page
            print(f"[INFO] Handling {link}")

            driver.set_page_load_timeout(30)
            driver.get(link)
            time.sleep(3)  # Basic wait for JS to render, as Yahoo Finance does not directly provide HTML with the needed content 

            # Parse HTML and extract content
            # TODO: Use BeautifulSoup html.parser to extract "content" from the loaded page, only targeting at <p> elements.
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            paragraphs = soup.find_all('p')
            content = '\n'.join([p.get_text() for p in paragraphs]).strip()

            # TODO: Validate if the content is emply; if yes, please continue current loop.
            if not content:
                continue
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)[:100]
            date_format = published.strftime('%Y-%m-%d %H:%M') # published is a datetime object you should get from entity

            # TODO: Append results to the news_list.
            news_list.append({
                'ticker': ticker_symbol,
                'title': title,
                'link': link,
                'date': published.strftime('%Y-%m-%d %H:%M'),
                'content': content
            })
        except Exception as e:
            print(f"[ERROR] Failed to fetch: {title}")
            print(f"[ERROR] Error: {e}")
            continue

    return news_list



# 13 pts
def save_news_to_txt(news_list, output_folder='stocknews'):
    """
    Save each news item to an individual text file organized in subfolders per ticker.

    Folder structure:
        stocknews/
            TICKER/
                news1.txt
                news2.txt
                ...

    Each file contains:
        Title
        Date
        URL
        Content

    Parameters:
        news_list (list[dict]): List of news dictionaries containing news details.
        output_folder (str): Root folder for saving news files.

    Instructions:
    1. Create the root output folder if it doesn't exist.
    2. For each news item, create a subfolder for the ticker if it doesn't exist.
    3. Construct a filename based on the date and title, ensuring it's valid for the filesystem.
    4. Write the news content to the text file in the specified format.
    """

    # TODO: Create the root output folder if it doesn't exist.
    os.makedirs(output_folder, exist_ok=True)
    
    # TODO: Loop through each news article in news_list.
        # TODO: Create a subfolder for the ticker if it doesn't exist.
        # TODO: Construct a filename based on the date and title, ensuring it's valid for the filesystem.
        # TODO: Write the news content to the text file in the specified format (see above function comment).
    for news in news_list:
        ticker_folder = os.path.join(output_folder, news['ticker'])
        os.makedirs(ticker_folder, exist_ok=True)

        safe_title = re.sub(r'[\\/*?:"<>|]', "_", news['title'])[:100]
        filename = f"{news['date'].replace(':', '-')}_{safe_title}.txt"
        filepath = os.path.join(ticker_folder, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {news['title']}\n")
            f.write(f"Date: {news['date']}\n")
            f.write(f"URL: {news['link']}\n")
            f.write("Content:\n")
            f.write(news['content'])

    print(f"[INFO] Saved {len(news_list)} news articles to '{output_folder}/'")

# 12 pts
def fetch_and_save_news_for_all_stocks(ticker_list, within_days):
    """
    Fetch news headlines for all tickers in the list and save them to text files.

    Parameters:
        ticker_list (list[str]): List of stock tickers to fetch news for.
        within_days (int): Limit news to recent headlines only, based on the number of days.

    Returns:
        None
    
    Instructions:
    1. Set up the Chrome WebDriver in headless mode to avoid opening a browser window.
    2. For each ticker in the provided list, call the function to fetch news articles.
    3. Collect and save all news articles in a single list.
    """
    # Setup Chrome in headless mode
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    # TODO: Create a WebDriver instance with the specified options.

    # TODO: Loop through each ticker in ticker_list.
    # TODO: Call fetch_news_for_one_stock for each ticker and store the results.
    # TODO: Use save_news_to_txt to save the fetched news articles.

    # TODO: Ensure the WebDriver is properly closed after all tickers have been processed.
    driver = webdriver.Chrome(options=options)

    try:
        all_news = []
        for ticker in ticker_list:
            articles = fetch_news_for_one_stock(ticker, driver, within_days)
            all_news.extend(articles)
        save_news_to_txt(all_news)
    finally:
        driver.quit()

# 0 pt
def test_submission():
    """
    Test runner for the news scraping task. This function fetches and saves news articles
    for a predefined list of stock tickers within a specified time frame and triggers file output.

    Optional Modifications:
    1. Change the list of tickers to test different stocks.
    2. Adjust the within_days parameter to fetch news from a different time frame.
    """
    tickers = ['XOM', 'CVX', 'HAL', 'MMM', 'CAT', 'DAL', 'MCD', 'NKE', 'KO',
               'JNJ', 'PFE', 'UNH', 'JPM', 'GS', 'BAC', 'AAPL', 'MSFT',
               'NVDA', 'GOOGL', 'META']
    within_days = 3  # News from the past 3 day
    fetch_and_save_news_for_all_stocks(tickers, within_days=within_days)



if __name__ == '__main__':
    test_submission()