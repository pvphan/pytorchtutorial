import torch


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
