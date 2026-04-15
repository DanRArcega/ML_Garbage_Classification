import torch
import torch.nn as nn
import torch.nn.functional as F

class GarbageClassificationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        
        self.pool  = nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        
        # 128 channels * 5 * 5 = 3200
        self.fc1   = nn.Linear(3200, 256)
        self.fc2   = nn.Linear(256, 128)
        self.fc3   = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 6)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        
        return self.fc4(x)