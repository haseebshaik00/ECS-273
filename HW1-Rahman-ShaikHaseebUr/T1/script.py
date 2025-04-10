import yfinance as yf
import pandas as pd
import os


# 12 pts
def fetch_one_stock(ticker_symbol, period='1mo'):
    """
    Fetch historical stock data for a single ticker using yfinance.

    Parameters:
        ticker_symbol (str): The stock symbol (e.g., 'NVDA').
        period (str): Time period to fetch (e.g., '1mo', '6mo').

    Returns:
        pd.DataFrame: DataFrame containing columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'].
    """
    # TODO: fetch data using yfinance
    #       return a DataFrame with the required columns

    return pd.DataFrame()  # replace with actual DataFrame


# 8 pts
def fetch_stocks(ticker_list, period='1mo'):
    """
    Fetch stock data for a list of tickers.

    Parameters:
        ticker_list (list[str]): List of stock symbols.
        period (str): Time period for each stock.

    Returns:
        dict: Dictionary mapping each ticker symbol to its DataFrame of stock data.
    """
    # TODO: call fetch_one_stock for each ticker in the list
    #       store the result in a dictionary as {ticker: dataframe}

    return {}  # replace with actual dictionary


# 8 pts
def save_stocks_to_csv(stock_data_dict, folder='stockdata'):
    """
    Save stock data to individual CSV files.

    Parameters:
        stock_data_dict (dict): Dictionary of {ticker: DataFrame}.
        folder (str): Folder where CSV files should be saved.

    Returns:
        None
    """
    # TODO: create the output folder if it does not exist
    #       save each DataFrame to a file named <ticker>.csv

    pass  # replace with actual saving logic


# 2 pts
def test_submission():
    """
    Test runner for stock data collection.

    You may modify this function for local testing, but the autograder will use it as the main entry point.
    Do not add code outside this function or outside __main__.
    """
    # TODO: expand the ticker list if needed
    tickers = ['NVDA', 'AAPL']
    # TODO: expand to the right period if needed
    period = '1mo'

    stock_data = fetch_stocks(tickers, period)
    save_stocks_to_csv(stock_data)


if __name__ == '__main__':
    test_submission()