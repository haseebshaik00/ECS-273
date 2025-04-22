import os
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import feedparser
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re

def fetch_news_for_one_stock(ticker_symbol, driver, within_days):
    news_list = []
    cutoff_date = datetime.now() - timedelta(days=within_days)

    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_symbol}&region=US&lang=en-US"
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        print(f"[WARNING] No entries found in RSS feed for {ticker_symbol}")
        return []

    print(f"[INFO] Processing news for {ticker_symbol} (last {within_days} days)")

    irrelevant_phrases = {
        "Oops, something went wrong",
        "Tip: Try a valid symbol or a specific company name for relevant results",
        "Sign in to access your portfolio",
        "Try again."
    }

    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6])
        if published < cutoff_date:
            continue

        title = entry.title
        link = entry.link

        try:
            print(f"[INFO] Handling {link}")
            driver.set_page_load_timeout(30)
            driver.get(link)
            time.sleep(3)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            paragraphs = soup.find_all('p')
            cleaned_lines = []
            for p in paragraphs:
                line = p.get_text(strip=True)
                if line and line not in irrelevant_phrases:
                    cleaned_lines.append(line)

            content = '\n'.join(cleaned_lines).strip()
            if not content:
                continue

            safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)[:100]
            date_format = published.strftime('%Y-%m-%d %H:%M')
            news_list.append({
                'ticker': ticker_symbol,
                'title': safe_title,
                'link': link,
                'date': date_format,
                'content': content
            })

        except Exception as e:
            print(f"[ERROR] Failed to fetch: {title}")
            print(f"[ERROR] Error: {e}")
            continue
    return news_list

def save_news_to_txt(news_list, output_folder='stocknews'):
    os.makedirs(output_folder, exist_ok=True)
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

def fetch_and_save_news_for_all_stocks(ticker_list, within_days):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    try:
        all_news = []
        for ticker in ticker_list:
            articles = fetch_news_for_one_stock(ticker, driver, within_days)
            all_news.extend(articles)
        save_news_to_txt(all_news)
    finally:
        driver.quit()

def test_submission():
    tickers = ['XOM', 'CVX', 'HAL', 'MMM', 'CAT', 'DAL', 'MCD', 'NKE', 'KO',
               'JNJ', 'PFE', 'UNH', 'JPM', 'GS', 'BAC', 'AAPL', 'MSFT',
               'NVDA', 'GOOGL', 'META']
    within_days = 3
    fetch_and_save_news_for_all_stocks(tickers, within_days=within_days)

if __name__ == '__main__':
    test_submission()