import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAudioCNN(nn.Module):
    """
    Basic convolutional neural network for classifying mel spectrograms.
    Expects input of shape (batch, 1, n_mels, time).
    Returns logits of shape (batch, num_classes).
    """
    def __init__(self, n_mels=128, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)
        # Fully connected layers are initialized after the first forward pass
        self.fc1 = None
        self.fc2 = None
        self.n_classes = n_classes

    def forward(self, x):
        # Input: (batch, 1, n_mels, time)
        assert x.ndim == 4, f"Expected 4D input (batch, 1, n_mels, time), got {x.shape}"
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        # Initialize fully connected layers on first pass
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
            self.fc2 = nn.Linear(128, self.n_classes).to(x.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def print_model_summary(model, input_shape=(1, 1, 128, 128)):
    """
    Print the model architecture and output shape for a dummy input.
    """
    x = torch.randn(input_shape)
    print(f"Input shape: {x.shape}")
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    print(model)

# Example usage:
# model = SimpleAudioCNN(n_mels=128, n_classes=4)
# print_model_summary(model, input_shape=(2, 1, 128, 128))
