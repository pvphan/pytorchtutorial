import numpy as np
import torch

import mnistdataset

imageSize = (28, 28)
inputSize = imageSize[0] * imageSize[1]
outputSize = 10

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


def initializeDevice(model):
    shouldUseCuda = torch.cuda.is_available()
    device = torch.device("cuda" if shouldUseCuda else "cpu")
    model.to(device)
    if shouldUseCuda:
        model.cuda()
    return device


def createTensor(array, device, inputSize):
    return torch.tensor(array, device=device, dtype=torch.float32).view(-1, inputSize)


def createLabelsArray(y, outputSize):
    """
    input:
        y -- (N,) integer labels
    output:
        labelArray -- (N, 10) consisting of zeros except where integer labels
                assign a 1.0
    """
    labelsArray = np.zeros((y.shape[0], outputSize))
    labelsArray[np.arange(y.shape[0]),y] = 1.0
    return labelsArray


def trainModel(model, inputTensorTrain, labelTensorTrain, learningRate, numEpochs):
    losses = []
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    for epoch in range(numEpochs):
        optimizer.zero_grad()
        outputsTensor = model(inputTensorTrain)

        loss = criterion(outputsTensor, labelTensorTrain)
        lossValue = loss.cpu().data.numpy()
        losses.append(lossValue)

        loss.backward()

        optimizer.step()
        print(f'epoch {epoch}, loss {loss.item()}')
    return losses


def main():
    modelClass, numEpochs, learningRate = MnistModelLinear, 10_000, 0.1 # 94%
    model = modelClass()
    device = initializeDevice(model)

    datasetDict = mnistdataset.loadDataset()
    x = datasetDict["train"]["images"]
    y = datasetDict["train"]["labels"]
    inputTensorTrain = createTensor(x / 255, device, inputSize)
    labelsArray = createLabelsArray(y, outputSize)
    labelTensorTrain = createTensor(labelsArray, device, outputSize)

    losses = trainModel(model, inputTensorTrain, labelTensorTrain, learningRate, numEpochs)

    xtest = datasetDict["test"]["images"]
    ytest = datasetDict["test"]["labels"]
    labelTensorTest = createTensor(xtest / 255, device, inputSize)
    with torch.no_grad(): # we don't need gradients in the testing phase
        predictedArray = model(labelTensorTest).cpu().data.numpy()
        predictedLabels = np.argmax(predictedArray, axis=1)
        numCorrectlyPredicted = np.sum(predictedLabels == ytest)
        print(f"Correctly predicted {100 * numCorrectlyPredicted/predictedLabels.shape[0]:0.2f}%")


if __name__ == "__main__":
    main()

