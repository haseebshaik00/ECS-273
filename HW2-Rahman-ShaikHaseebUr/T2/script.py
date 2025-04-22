import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from utils import fetch_stocks, save_stocks_to_csv # the function you implemented in HW1/T1

# 5 pts
class StockDataset(Dataset):
    def __init__(self, folder_path = 'stockdata', feature_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
        """_summary_

        Args:
            folder_path (str, optional): _description_. Defaults to 'stockdata'.
            feature_cols (list, optional): _description_. Defaults to ['Open', 'High', 'Low', 'Close', 'Volume'].
        """
        # TODO: load all CSV files from the folder
        #       concatenate them into a single DataFrame
        #       any normalzation or other preprocessing should be done here

    def __len__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # TODO: return length of the input dataset
        return 0 # replace with actual length

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        # TODO: return the stock data for the given index
        return 0 # replace with actual data

# 30 pts
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, latent_dim=16, seq_len=23):
        """_summary_

        Args:
            input_dim (int, optional): _description_. Defaults to 5.
            hidden_dim (int, optional): _description_. Defaults to 64.
            latent_dim (int, optional): _description_. Defaults to 16.
            seq_len (int, optional): _description_. Defaults to 23.
        """
        super().__init__()
        # TODO: define the LSTM encoder and decoder
    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # TODO: implement the forward pass
        #       return the reconstructed input and the latent representation
        return 0, 0 # replace with actual output

# 10 pts
def train_autoencoder(model, dataloader, num_epochs=50, lr=5e-3):
    """_summary_

    Args:
        model (_type_): _description_
        dataloader (_type_): _description_
        num_epochs (int, optional): _description_. Defaults to 50.
        lr (_type_, optional): _description_. Defaults to 5e-3.
    """
    # TODO: define the training process

# 5 pts
def get_latent(model, dataloader):
    """_summary_

    Args:
        model (_type_): _description_
        dataloader (_type_): _description_

    Returns:
        _type_: _description_
    """
    # TODO: Apply the model evaluation mode and return the latent representations
    return 0 # replace with actual latent representations

# 0 pts 
# For testing your functions and classes, you can modify this based on your needs.
def test_submission():
    tickers = ['XOM', 'CVX', 'HAL'] # replace with the tickers you want to test
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

if __name__ == '__main__':
    test_submission()