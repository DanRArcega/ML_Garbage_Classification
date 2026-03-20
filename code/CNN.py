import torch
import torch.nn as nn
import torch.nn.functional as F

# This is a basic skeleton that will be revamped once development on the CNN starts
class GarbageClassificationCNN(nn.Module):
    def __init__(self):
        # Use the default architecture from the PyTorch tutorial as a starting point
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input channel (grayscale)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))  # Add adaptive pooling to ensure the output size is consistent regardless of input image size
        self.fc1   = nn.Linear(16 * 5 * 5, 120) 
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
