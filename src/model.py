# src/model.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layer with fewer output channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5)
        # Batch Normalization after convolution
        # self.bn1 = nn.BatchNorm2d(num_features=2)
        # Increase kernel size in pooling to reduce dimensions more
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.fc1 = nn.Linear(in_features=2 * 8 * 8, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)  # Apply batch normalization
        x = nn.functional.relu(x)  # Apply ReLU activation
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# Calculate parameter size
if __name__ == "__main__":
    model = SimpleCNN()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model.conv1(dummy_input)
    print(f"Output shape after conv1: {output.shape}")  # Expected: [1, 2, 24, 24]
    # output = model.bn1(output)
    # print(f"Output shape after batch normalization: {output.shape}")  # Should be the same
    output = nn.functional.relu(output)
    output = model.pool(output)
    print(f"Output shape after pooling: {output.shape}")  # Expected: [1, 2, 8, 8]
    flattened_output = model.flatten(output)
    print(f"Shape after flattening: {flattened_output.shape}")  # Expected: [1, 128]
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_size_kbits = (param_size_bytes * 8) / 1024
    print(f"Model parameter size: {param_size_kbits:.2f} Kbits")