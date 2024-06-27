import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn(nn.Module):
    def __init__(self) -> None:
        super(cnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1) # input of 1 channel, output 32 convolutional features, kernel size 3x3, stride 1
        self.pool1 = nn.MaxPool2d(2, 2) # 2x2 pooling (max pooling), stride 2 (non-overlapping pooling)

        self.conv2 = nn.Conv2d(32, 64, 3, 1) # input of 32 channel, output 64 convolutional features, kernel size 3x3, stride 1
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128*3*3, 128)
    
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

        return x


random_data = torch.rand((1, 1, 28, 28))

my_nn = cnn()
result = my_nn(random_data)
print(result.size())