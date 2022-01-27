import os

def loadDataset():
    datasetPath = "/pytorchtutorial/data/"
    datasetDict = {
        "train": {
            "images": loadImages(f"{datasetPath}/train-images-idx3-ubyte"),
            "labels": loadLabels(f"{datasetPath}/train-labels-idx1-ubyte"),
        },
        "test": {
            "images": loadImages(f"{datasetPath}/t10k-images-idx3-ubyte"),
            "labels": loadLabels(f"{datasetPath}/t10k-labels-idx1-ubyte"),
        },
    }
    return datasetDict


def loadImages(imagesFilePath):
    raise NotImplementedError()


def loadLabels(labelsFilePath):
    raise NotImplementedError()

