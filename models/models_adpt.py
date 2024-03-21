import torch
import torch.nn as nn
from torch.nn import functional as F
from models.ABN import MultiBatchNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class classifier_spk(nn.Module):
    def __init__(self, num_classes=15):
        super(classifier_spk, self).__init__()
        self.num_classes = num_classes

        # Define batch normalization and ReLU activation
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(512, 512, bias=False)

        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.ReLU()
        self.fc1_1 = nn.Linear(512, 256, bias=False)

        self.bn1_2 = nn.BatchNorm1d(256)
        self.relu1_2 = nn.ReLU()
        self.fc1_2 = nn.Linear(256, 256, bias=False)

        self.fc2 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x, return_feature=False):
        # Flatten the input
        x = torch.flatten(x, 1)
        
        # Apply the first set of layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Apply the second set of layers
        x = self.fc1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)


        # Apply the third set of layers
        x = self.fc1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        # Apply the final fully connected layer to produce logits for classification
        y = self.fc2(x)

        # If return_feature is True, return both the feature and the output
        if return_feature:
            return x, y
        else:
            return y