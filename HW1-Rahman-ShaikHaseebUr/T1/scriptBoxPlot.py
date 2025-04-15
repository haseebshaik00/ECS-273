import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# def fetch_one_stock(ticker_symbol, period='1mo'):
#     df = yf.download(ticker_symbol, period=period, group_by='ticker')
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
#     df.reset_index(inplace=True)
#     cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
#     df = df[[col for col in df.columns if any(c in col for c in cols)]]
#     df.columns = cols
#     return df

# def fetch_stocks(ticker_list, period='1mo'):
#     return {ticker: fetch_one_stock(ticker, period) for ticker in ticker_list}

# def save_stocks_to_csv(stock_data_dict, folder='stockdata'):
#     os.makedirs(folder, exist_ok=True)
#     for ticker, df in stock_data_dict.items():
#         df.to_csv(f"{folder}/{ticker}.csv", index=False)

# def test_submission():
#     tickers = ['XOM', 'CVX', 'HAL', 'MMM', 'CAT', 'DAL', 'MCD', 'NKE', 'KO',
#                'JNJ', 'PFE', 'UNH', 'JPM', 'GS', 'BAC', 'AAPL', 'MSFT',
#                'NVDA', 'GOOGL', 'META']
#     period = '2y'
#     stock_data = fetch_stocks(tickers, period)
#     save_stocks_to_csv(stock_data)

def generate_stock_summary_and_plot(data_folder='stockdata', output_folder='.'):
    os.makedirs(output_folder, exist_ok=True)

    summary_data = []
    plot_data = []

    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            ticker = filename.replace('.csv', '')
            filepath = os.path.join(data_folder, filename)
            df = pd.read_csv(filepath)
            mean_close = round(df['Close'].mean(), 2)
            std_close = round(df['Close'].std(), 2)
            summary_data.append({'Ticker': ticker, 'Mean_Close': mean_close, 'SD_Close': std_close})
            df['Ticker'] = ticker
            plot_data.append(df[['Ticker', 'Close']])

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Ticker')
    summary_df.to_csv(os.path.join(output_folder, 'summary_table.csv'), index=False)
    combined_df = pd.concat(plot_data)
    plt.figure(figsize=(16, 6))
    sns.boxplot(data=combined_df, x='Ticker', y='Close')
    plt.title('Distribution of Closing Prices (Last 2 Years)')
    plt.xlabel('Stock Ticker')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'stock_boxplot.png')
    plt.savefig(plot_path)
    plt.close()

if __name__ == '__main__':
    generate_stock_summary_and_plot()