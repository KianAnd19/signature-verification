import torch
import torch.nn as nn
import torch.nn.functional as F

class snn(nn.Module):
    def __init__(self) -> None:
        super(snn, self).__init__()
        self.cnn = cnn()
        self.fc1 = nn.Linear(128, 1)
        
    def forward(self, x, y):
        result1 = self.cnn(x)
        result2 = self.cnn(y)
        
        z = torch.abs(result1 - result2)  # absolute difference
        
        z = self.fc1(z)
        return z


class cnn(nn.Module):
    def __init__(self) -> None:
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)  # Added padding to maintain spatial dimensions
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # With 32x32 input, after 3 pooling layers (2x2), we get 4x4 feature maps
        # So 128 channels * 4 * 4 = 2048 features
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc1(x)
        return x

