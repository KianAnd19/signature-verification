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
        z = torch.sigmoid(z)
        return z


class cnn(nn.Module):
    def __init__(self) -> None:
        super(cnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # input of 1 channel, output 32 convolutional features, kernel size 3x3, stride 1
        self.pool1 = nn.MaxPool2d(2, 2)  # 2x2 pooling (max pooling), stride 2 (non-overlapping pooling)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # input of 32 channel, output 64 convolutional features, kernel size 3x3, stride 1
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.fc_input_size = self._get_conv_output((1, 200, 200))
        self.fc1 = nn.Linear(self.fc_input_size, 128)
    
    def _get_conv_output(self, shape):
        # This function calculates the size of the output tensor after the convolutional layers
        with torch.no_grad():
            input = torch.autograd.Variable(torch.ones(1, *shape))
            output_feat = self._forward_features(input)
            n_size = output_feat.data.view(1, -1).size(1)
        return n_size
    
    def _forward_features(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc1(x)
        return x
