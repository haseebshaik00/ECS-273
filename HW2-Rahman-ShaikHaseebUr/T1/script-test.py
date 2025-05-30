import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from script import Model, prediction, load_mnist

_, test_loader = load_mnist()

model = Model()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

y_pred = prediction(model, test_loader)
y_true = []
for _, labels in test_loader:
    y_true.extend(labels.tolist())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title("Confusion Matrix - MNIST Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True/Actual Label")
plt.tight_layout()
plt.savefig("./confusion_matrix.png")
plt.show()