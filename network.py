import torch
import torch.nn as nn
import torch.nn.functional as F

class snn(nn.Module):
    def __init__(self) -> None:
        super(snn, self).__init__()
        
        self.fc1 = nn.Linear(128, 1)
        
    def forward(self, x, y):
        cnn1 = cnn()
        cnn2 = cnn()
        result1 = cnn1(x)
        result2 = cnn2(y)
        
        z = result1 - result2
        
        z = self.fc1(z)
        z = F.sigmoid(z)
        return z


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
        x = x.view(x.size(0), -1) # flatten the tensor
        return x


random_data1 = torch.rand((1, 1, 28, 28))
random_data2 = torch.rand((1, 1, 28, 28))

my_nn = snn()
result = my_nn(random_data1, random_data2)
print(result)