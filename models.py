import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(torch.nn.Module):
    # https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
    def __init__(self, numClasses: int):
        super().__init__()
        self._featureExtractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self._classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=numClasses),
        )

    def forward(self, x):
        x = self._featureExtractor(x)
        x = torch.flatten(x, 1)
        logits = self._classifier(x)
        probabilities = F.softmax(logits, dim=1)
        return logits, probabilities


class MnistModelLinear(torch.nn.Module):
    def __init__(self):
        """
        n_out = ((n_in + 2p - k) / s) + 1
        n_out   -- number of output features
        n_in    -- number of input features
        k       -- convolution kernel size
        p       -- convolution padding size
        s       -- convolution stride size
        """
        super(MnistModelLinear, self).__init__()
        baseHiddenLayerSize = 16
        inputSize = 28 * 28
        outputSize = 10

        self.hiddenLayer1 = torch.nn.Linear(inputSize, 2 * baseHiddenLayerSize)
        self.hiddenLayer2 = torch.nn.Linear(2 * baseHiddenLayerSize, baseHiddenLayerSize)
        self.hiddenLayer3 = torch.nn.Linear(baseHiddenLayerSize, outputSize)
        self.relu = torch.nn.ReLU()

        self.fullNetworkFunction = torch.nn.Sequential(
            self.hiddenLayer1,
            self.relu,
            self.hiddenLayer2,
            self.relu,
            self.hiddenLayer3,
        )

    def forward(self, x):
        return self.fullNetworkFunction(x)
