import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from utils_HW1 import fetch_stocks, save_stocks_to_csv

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
        """_summary_
        Returns:
            _type_: _description_
        """
        # # TODO: return length of the input dataset
        # return 0 # replace with actual length
        return len(self.data)

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        # # TODO: return the stock data for the given index
        # return 0 # replace with actual data
        return self.data[idx]

# 30 pts
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, latent_dim=16, seq_len=23):
        """
        Args:
            input_dim (int, optional): _description_. Defaults to 5.
            hidden_dim (int, optional): _description_. Defaults to 64.
            latent_dim (int, optional): _description_. Defaults to 16.
            seq_len (int, optional): _description_. Defaults to 23.
        """
        super().__init__()
        # TODO: define the LSTM encoder and decoder
        # super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # # TODO: implement the forward pass
        # #       return the reconstructed input and the latent representation
        # return 0, 0 # replace with actual output
        batch_size = x.size(0)
        enc_out, _ = self.encoder(x)
        latent = self.fc_enc(enc_out[:, -1, :])  # take last hidden state

        # Repeat latent vector across sequence length
        repeated = self.fc_dec(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder(repeated)
        return dec_out, latent


# 10 pts
def train_autoencoder(model, dataloader, num_epochs=50, lr=5e-3):
    """
    Args:
        model (_type_): _description_
        dataloader (_type_): _description_
        num_epochs (int, optional): _description_. Defaults to 50.
        lr (_type_, optional): _description_. Defaults to 5e-3.
    """
    # TODO: define the training process
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

# 5 pts
def get_latent(model, dataloader):
    """
    Args:
        model (_type_): _description_
        dataloader (_type_): _description_
    Returns:_type_: _description_
    """
    # TODO: Apply the model evaluation mode and return the latent representations
    # return 0 # replace with actual latent representations
    model.eval()
    latent_reps = []

    with torch.no_grad():
        for batch in dataloader:
            _, latent = model(batch)
            latent_reps.append(latent)
    return torch.cat(latent_reps, dim=0)

# 0 pts 
# For testing your functions and classes, you can modify this based on your needs.
def test_submission():
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
    torch.save(model.state_dict(), "lstm_model.pth")

if __name__ == '__main__':
    test_submission()