import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_news_articles(input_folder='stocknews', output_folder='.'):
    os.makedirs(output_folder, exist_ok=True)

    summary_data = []
    boxplot_data = []

    for ticker in os.listdir(input_folder):
        ticker_path = os.path.join(input_folder, ticker)
        if not os.path.isdir(ticker_path):
            continue

        article_lengths = []

        for file in os.listdir(ticker_path):
            if file.endswith('.txt'):
                filepath = os.path.join(ticker_path, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Content:\n' in content:
                        body = content.split('Content:\n')[-1]
                        article_lengths.append(len(body.strip()))

        if article_lengths:
            mean_len = round(sum(article_lengths) / len(article_lengths), 2)
            std_len = round(pd.Series(article_lengths).std(), 2)
            summary_data.append({
                'Ticker': ticker,
                'Mean_Char_Length': mean_len,
                'Std_Char_Length': std_len
            })
            for length in article_lengths:
                boxplot_data.append({'Ticker': ticker, 'Length': length})

    # Save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Ticker')
    summary_df.to_csv(os.path.join(output_folder, 'news_summary_table.csv'), index=False)

    # Generate boxplot
    plot_df = pd.DataFrame(boxplot_data)
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=plot_df, x='Ticker', y='Length')
    plt.title('Distribution of News Article Lengths by Ticker')
    plt.xlabel('Stock Ticker')
    plt.ylabel('Article Character Length')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'news_boxplot.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Summary CSV saved to: {output_folder}/news_summary_table.csv")
    print(f"[INFO] Boxplot image saved to: {output_folder}/news_boxplot.png")

if __name__ == '__main__':
    analyze_news_articles()