import torch.nn.functional as F
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(12,12), stride=2, padding=0)
        # 1 channel in - grayscale. 25 out as 25 filters. 12x12x1 kernel. 2 stride in both directions. No padding. 
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=64, kernel_size=(5,5), stride=1, padding=2)
        self.pool = nn.MaxPool2d(2)  # Defines a max pool with a 2x2 square stride 2. 
        self.connected1 = nn.Linear(1024, 1024)  # Unsure if 1024 is the right output size
        self.dropout = nn.Dropout(p=0.2)
        self.connected2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x,start_dim=1)  # Don't want the batch to be flattened
        x = F.relu(self.connected1(x))
        x = self.dropout(x)
        x = self.connected2(x)
        return x