import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

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