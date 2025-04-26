import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from utils_HW1 import fetch_stocks, save_stocks_to_csv

class StockDataset(Dataset):
    def __init__(self, folder_path = 'stockdata', feature_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
        """
        - Summary: Custom PyTorch Dataset class to load and normalize stock time series data from 
                    CSV files within a specified folder
        - Args:
            folder_path (str): Path to the folder containing stock CSV files
            feature_cols (list): List of columns to use as features
        - Attributes: data (list): A list of normalized torch tensors for each stock
        - Implementation: Loads stock CSVs, selects key features, normalizes them (z-score), 
                            and converts them to tensors for model input
        """
        self.data = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, filename))
                if all(col in df.columns for col in feature_cols):
                    # Normalize each feature
                    df = df[feature_cols]
                    df = (df - df.mean()) / df.std()
                    self.data.append(df.values)
        # Convert each stock’s full history into a tensor
        self.data = [torch.tensor(stock, dtype=torch.float32) for stock in self.data]

    def __len__(self):
        """
        - Returns: int: Number of stock sequences in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        - Args: idx (int): Index of the item to fetch.
        - Returns: Tensor: Normalized stock time series tensor at given index.
        """
        return self.data[idx]

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, latent_dim=16, seq_len=23):
        """
        - Summary: LSTM Autoencoder model for learning latent representations of stock time series
        - Args:
            1) input_dim (int): Number of input features per timestep
            2) hidden_dim (int): Hidden state dimension for encoder and decoder LSTMs
            3) latent_dim (int): Dimension of the latent compressed representation
            4) seq_len (int): Length of the input sequence to reconstruct
        - Implementation: Encodes input sequences using LSTM, compresses to a latent vector, 
                            then decodes back to reconstruct the sequence
        """
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        """
        - Summary: Forward pass of the autoencoder
        - Args: x (Tensor): Input batch of shape (batch_size, seq_len, input_dim)
        - Returns:
            Tuple[Tensor, Tensor]: 
                1) Reconstructed input (same shape as x)
                2) Latent representation of the input (batch_size, latent_dim)
        - Implementation: Runs input through encoder to get latent vector, 
                            repeats and decodes it to reconstruct the original sequence
        """
        batch_size = x.size(0)
        enc_out, _ = self.encoder(x)
        latent = self.fc_enc(enc_out[:, -1, :])

        # Repeat latent vector across sequence length
        repeated = self.fc_dec(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder(repeated)
        return dec_out, latent


def train_autoencoder(model, dataloader, num_epochs=50, lr=5e-3):
    """
    - Summary: Trains the LSTM autoencoder on the given dataloader
    - Args:
        model (nn.Module): The LSTM autoencoder model
        dataloader (DataLoader): DataLoader providing input batches
        num_epochs (int): Number of training epochs
        lr (float): Learning rate for the Adam optimizer
    - Implementation: Trains the model using MSE loss between input and reconstruction, 
                        with Adam optimizer across multiple epochs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

def get_latent(model, dataloader):
    """
    - Summary: Extracts and returns the latent representations for each sequence
    - Args:
        model (nn.Module): Trained LSTM autoencoder
        dataloader (DataLoader): DataLoader with the input sequences
    - Returns: Tensor: A tensor of latent vectors for the input data
    - Implementation: Runs the model in eval mode and collects latent vectors for each input without computing gradients
    """
    model.eval()
    latent_reps = []

    with torch.no_grad():
        for batch in dataloader:
            _, latent = model(batch)
            latent_reps.append(latent)
    return torch.cat(latent_reps, dim=0)

def test_submission():
    """
    - Summary: End-to-end test function to fetch stock data, train the autoencoder,
                extract latent representations, and save the trained model
    - Implementation: Executes the full pipeline: fetches stock data, preprocesses it, trains the model, 
                        extracts latents, and saves weights.
    """
    tickers = ['XOM', 'CVX', 'HAL', 'MMM', 'CAT', 'DAL', 'MCD', 'NKE', 'KO',
               'JNJ', 'PFE', 'UNH', 'JPM', 'GS', 'BAC', 'AAPL', 'MSFT',
               'NVDA', 'GOOGL', 'META']
    period = '1mo'
    
    stock_data = fetch_stocks(tickers, period)
    save_stocks_to_csv(stock_data)
    
    dataset = StockDataset("stockdata")
    first_sample = dataset[0]
    
    seq_len = first_sample.shape[0]
    dataloader = DataLoader(dataset)

    model = LSTMAutoencoder(seq_len=seq_len)
    train_autoencoder(model, dataloader)

    Z = get_latent(model, dataloader)
    # Save the model
    torch.save(model.state_dict(), "lstm_model.pth")

if __name__ == '__main__':
    test_submission()