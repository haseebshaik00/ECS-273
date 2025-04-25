import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size=64):
    """
    Load MNIST training and test sets.
    Parameters:
        batch_size (int): Batch size for data loaders.
    Returns:
        tuple: (train_loader, test_loader)
    """
    # implement dataset loading and preprocessing
    # Use torchvision.datasets.MNIST
    # return loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class Model(nn.Module):
    """
    A simple neural network for MNIST classification.
    You should flatten the input and pass it through linear layers.
    """
    def __init__(self):
        super().__init__()
        # model structure
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10))

    def forward(self, x):
        # forward pass
        return self.network(x)

def train_one_epoch(model, dataloader, optimizer, loss_fn):
    """
    Parameters:
        model (nn.Module) : model to be trained
        dataloader (DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_fn: The loss function.
    Returns:
        loss (float): The average loss for the epoch.
    """
    # Training loop for one epoch
    # return actual loss
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
    
def prediction(model, dataloader):
    """
    Evaluate model accuracy on test set
    Parameters:
        model (nn.Module): The trained model
        dataloader (DataLoader): The test data loader
    Returns:
        predictions (list): List of predicted labels
    """
    # Evaluation logic
    # return actual predictions
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            predictions.extend(predicted.tolist())
    return predictions

def model_training(num_epochs=10):
    """
    Runs training for multiple epochs and prints evaluation results.
    You can modify this function, such as changing the number of epochs, learning rate, etc for your experiments.
    Parameters: num_epochs (int): Number of epochs to train.
    """
    train_loader, test_loader = load_mnist()
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch + 1}/{num_epochs} | Training Loss: {loss:.4f}")

    test_predictions = prediction(model, test_loader)
    true_labels = []
    for _, labels in test_loader:
        true_labels.extend(labels.tolist())

    # Model Accuracy
    correct = sum(p == t for p, t in zip(test_predictions, true_labels))
    accuracy = 100 * correct / len(true_labels)
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save model
    torch.save(model.state_dict(), "mnist_model.pth")
    
if __name__ == '__main__':
    model_training()