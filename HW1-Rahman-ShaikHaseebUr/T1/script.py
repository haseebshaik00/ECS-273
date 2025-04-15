import yfinance as yf
import pandas as pd
import os

def fetch_one_stock(ticker_symbol, period='1mo'):
    df = yf.download(ticker_symbol, period=period)
    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def fetch_stocks(ticker_list, period='1mo'):
    return {ticker: fetch_one_stock(ticker, period) for ticker in ticker_list}

def save_stocks_to_csv(stock_data_dict, folder='stockdata'):
    os.makedirs(folder, exist_ok=True)
    for ticker, df in stock_data_dict.items():
        df.to_csv(f"{folder}/{ticker}.csv", index=False)

def test_submission():
    tickers = ['XOM', 'CVX', 'HAL', 'MMM', 'CAT', 'DAL', 'MCD', 'NKE', 'KO',
               'JNJ', 'PFE', 'UNH', 'JPM', 'GS', 'BAC', 'APPL', 'MSFT',
               'NVDA', 'GOOGL', 'META']
    period = '2y'
    stock_data = fetch_stocks(tickers, period)
    save_stocks_to_csv(stock_data)

if __name__ == '__main__':
    test_submission()