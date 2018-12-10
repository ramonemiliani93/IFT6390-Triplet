"""Defines the models"""

import torch.nn as nn

# Constants
NUM_CLASSES = 10
IN_FEATURES = 784

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class LinearRegression(nn.Module):
    """
    Softmax regression model
    """

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.num_classes = NUM_CLASSES
        self.in_features = IN_FEATURES
        self.features = nn.Sequential(
            nn.Linear(self.in_features, self.num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return x


class MLP(nn.Module):
    """
    Multilayer perceptron model
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.num_classes = NUM_CLASSES
        self.in_features = IN_FEATURES
        # Add input layer and output layer
        self.features = nn.Sequential(
            nn.Linear(self.in_features, 50),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(50, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2, self.num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.bottleneck(x)
        return x

    def extract_features(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return x


class CNN(nn.Module):
    """
    CNN model
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.num_classes = NUM_CLASSES
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(inplace=True),
        )
        self.bottleneck =nn.Sequential(
            nn.Linear(50, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.bottleneck(x)
        return x

    def extract_features(self, x):
        x = self.features(x)
        return x