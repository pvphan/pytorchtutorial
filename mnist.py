import numpy as np
import torch

import mnistdataset

imageSize = (28, 28)
inputSize = imageSize[0] * imageSize[1]
outputSize = 10

class MnistModel(torch.nn.Module):
    def __init__(self):
        """
        n_out = ((n_in + 2p - k) / s) + 1

        n_out   -- number of output features
        n_in    -- number of input features
        k       -- convolution kernel size
        p       -- convolution padding size
        s       -- convolution stride size
        """
        super(MnistModel, self).__init__()
        hiddenLayerSize = 16

        self.hiddenLayer1 = torch.nn.Linear(inputSize, hiddenLayerSize)
        self.hiddenLayer2 = torch.nn.Linear(hiddenLayerSize, outputSize)
        self.relu = torch.nn.ReLU()

        self.fullNetworkFunction = torch.nn.Sequential(
            self.hiddenLayer1,
            self.relu,
            self.hiddenLayer2,
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
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputsTensor = model(inputTensorTrain)

        # get loss for the predicted output
        loss = criterion(outputsTensor, labelTensorTrain)
        lossValue = loss.cpu().data.numpy()
        losses.append(lossValue)

        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        print(f'epoch {epoch}, loss {loss.item()}')


def main():
    model = MnistModel()
    device = initializeDevice(model)
    learningRate = 0.1
    #numEpochs, learningRate = 1_000, 0.1 # 84.12%
    numEpochs, learningRate = 10_000, 0.1 # 91.97%
    #numEpochs, learningRate = 100_000, 0.1 # 93.46%

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
        print(f"correctly predicted {100 * numCorrectlyPredicted/predictedLabels.shape[0]:0.2f}%")


if __name__ == "__main__":
    main()
