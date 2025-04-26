import torch
import pandas as pd
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from script import StockDataset, LSTMAutoencoder, get_latent

ticker_to_sector = {
    'XOM': "Energy", 'CVX': "Energy", 'HAL': "Energy",
    'MMM': "Industrials", 'CAT': "Industrials", 'DAL': "Industrials",
    'MCD': "Consumer Discretionary/Staples", 'NKE': "Consumer Discretionary/Staples", 'KO': "Consumer Discretionary/Staples",
    'JNJ': "Healthcare", 'PFE': "Healthcare", 'UNH': "Healthcare",
    'JPM': "Financials", 'GS': "Financials", 'BAC': "Financials",
    'AAPL': "Information Tech / Comm. Svc", 'MSFT': "Information Tech / Comm. Svc",
    'NVDA': "Information Tech / Comm. Svc", 'GOOGL': "Information Tech / Comm. Svc", 'META': "Information Tech / Comm. Svc"
}

def load_labels(folder_path='stockdata'):
    """Creates a list of sectors as labels based on the tickers"""
    tickers = [f.replace(".csv", "") for f in os.listdir(folder_path) if f.endswith('.csv')]
    return [ticker_to_sector[ticker] for ticker in tickers]

def tsne_and_plot(data, labels, title, filename):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced = tsne.fit_transform(data)
    df = pd.DataFrame({
        'X': reduced[:, 0],
        'Y': reduced[:, 1],
        'Label': labels
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='X', y='Y', hue='Label', palette='tab10', s=100)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def main():
    dataset = StockDataset("stockdata")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    labels = load_labels("stockdata")
    seq_len = dataset[0].shape[0]
    model = LSTMAutoencoder(seq_len=seq_len)
    model.load_state_dict(torch.load("lstm_model.pth"))
    latent_reps = get_latent(model, dataloader).numpy()

    tsne_and_plot(
        latent_reps,
        labels=labels,
        title="Latent Representation- Grouped by Category/Sector",
        filename="tsne_latent.png")

    raw_data = np.array([x.numpy().flatten() for x in dataset])
    tsne_and_plot(
        raw_data,
        labels=labels,
        title="Raw Time Series- Grouped by Category/Sector",
        filename="tsne_raw.png")

if __name__ == '__main__':
    main()
