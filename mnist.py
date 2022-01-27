import numpy as np
import torch

import mnistdataset


class MnistModel(torch.nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        imageSize = (28, 28)
        hiddenLayerSize = 16
        outputSize = 10

        imageVectorSize = imageSize[0] * imageSize[1]
        self.hiddenLayer1 = torch.nn.Conv1d(1, 1, imageVectorSize)
        self.hiddenLayer2 = torch.nn.Conv1d(1, 1, hiddenLayerSize)
        self.outputLayer = torch.nn.Conv1d(1, 1, outputSize)
        self.relu = torch.nn.ReLU()

        self.fullNetworkFunction = torch.nn.Sequential(
            self.hiddenLayer1,
            self.relu,
            self.hiddenLayer2,
            self.relu,
            self.outputLayer,
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


def main():
    model = MnistModel()
    device = initializeDevice(model)
    learningRate = 0.1
    epochs = 100

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    datasetDict = mnistdataset.loadDataset()
    x = datasetDict["train"]["images"]
    y = datasetDict["train"]["labels"]
    inputs = torch.tensor(x, device=device, dtype=torch.float32).view(-1,1)
    labels = torch.tensor(y, device=device, dtype=torch.float32).view(-1,1)

    losses = []

    for epoch in range(epochs):
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        print(loss)

        # get gradients w.r.t to parameters
        loss.backward()

        losses.append(loss.detach().cpu().numpy())

        # update parameters
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))


    with torch.no_grad(): # we don't need gradients in the testing phase
        predicted = model(inputs).cpu().data.numpy()
        print(predicted)


if __name__ == "__main__":
    main()
