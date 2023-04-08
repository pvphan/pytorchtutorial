# https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

import mnist
import mnistdataset
import models


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    train_losses = []
    valid_losses = []
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
              )

    return model, optimizer, (train_losses, valid_losses)


def train(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def main():
    numEpochs = 2
    learningRate = 0.001
    numClasses = 10
    model = models.LeNet5(numClasses)
    device = mnist.initializeDevice(model)

    randomSeed = 0
    torch.manual_seed(randomSeed)

    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss()
    dataRoot = "/tmp/data"
    batchSize = 32
    trainDataLoader, valDataLoader = mnistdataset.getDataLoaders(dataRoot, batchSize)
    model, optimizer, _ = training_loop(
            model, criterion, optimizer, trainDataLoader, valDataLoader, numEpochs, device)

    model.eval()
    allLabels = []
    allPredictions = []
    for X, labels in valDataLoader:
        X = X.to(device)
        allLabels.extend(mnist.tensorToNumpy(labels))
        labels = labels.to(device)

        _, y_probs = model(X)
        predictions = mnist.tensorToNumpy(torch.argmax(y_probs, dim=1))
        allPredictions.extend(predictions)
    numCorrectlyPredicted = np.sum(np.array(allPredictions) == np.array(allLabels))
    print(f"Error rate: {100 - 100 * numCorrectlyPredicted/len(allLabels):0.2f}%")


if __name__ == "__main__":
    main()
