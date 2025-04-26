# import torch
# import pandas as pd
# import os
# import numpy as np  # Make sure it's imported at the top if not already
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import seaborn as sns
# from script import StockDataset, LSTMAutoencoder, get_latent

# def load_labels(folder_path='stockdata'):
#     """Creates a list of tickers as labels based on the filenames"""
#     return [f.replace(".csv", "") for f in os.listdir(folder_path) if f.endswith('.csv')]

# def tsne_and_plot(data, labels, title, filename):
#     tsne = TSNE(n_components=2, random_state=42, perplexity=5)
#     reduced = tsne.fit_transform(data)

#     df = pd.DataFrame({
#         'X': reduced[:, 0],
#         'Y': reduced[:, 1],
#         'Label': labels
#     })

#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(data=df, x='X', y='Y', hue='Label', palette='tab10')
#     plt.title(title)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.show()

# def main():
#     # Load dataset and dataloader
#     dataset = StockDataset("stockdata")
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

#     labels = load_labels("stockdata")

#     # === 1. t-SNE on latent representations ===
#     seq_len = dataset[0].shape[0]
#     model = LSTMAutoencoder(seq_len=seq_len)
#     model.load_state_dict(torch.load("mnist_model.pth"))  # Replace with your trained model path
#     latent_reps = get_latent(model, dataloader).numpy()

#     tsne_and_plot(
#         latent_reps,
#         labels=labels,
#         title="t-SNE on Latent Representations",
#         filename="tsne_latent.png"
#     )

#     # === 2. t-SNE on raw time series (flattened) ===
#     # raw_data = [x.numpy().flatten() for x in dataset]
#     raw_data = np.array([x.numpy().flatten() for x in dataset])
#     tsne_and_plot(
#         raw_data,
#         labels=labels,
#         title="t-SNE on Raw Time Series",
#         filename="tsne_raw.png"
#     )

# if __name__ == '__main__':
#     main()

import torch
import pandas as pd
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from script import StockDataset, LSTMAutoencoder, get_latent

# Ticker → Sector mapping
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
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(features) - 1))
    reduced = tsne.fit_transform(data)
    df = pd.DataFrame({
        'X': reduced[:, 0],
        'Y': reduced[:, 1],
        'Label': labels
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='X', y='Y', hue='Label', palette='tab10', s=100)  # Increase marker size
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def main():
    dataset = StockDataset("stockdata")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Get sector labels
    labels = load_labels("stockdata")

    # === 1. t-SNE on latent representations ===
    seq_len = dataset[0].shape[0]
    model = LSTMAutoencoder(seq_len=seq_len)
    model.load_state_dict(torch.load("mnist_model.pth"))  # Replace if needed
    latent_reps = get_latent(model, dataloader).numpy()

    tsne_and_plot(
        latent_reps,
        labels=labels,
        title="t-SNE on Latent Representations (Colored by Sector)",
        filename="tsne_latent.png"
    )

    # === 2. t-SNE on raw time series (flattened) ===
    raw_data = np.array([x.numpy().flatten() for x in dataset])
    tsne_and_plot(
        raw_data,
        labels=labels,
        title="t-SNE on Raw Time Series (Colored by Sector)",
        filename="tsne_raw.png"
    )

if __name__ == '__main__':
    main()
