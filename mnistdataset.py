import os
import re
import struct
from typing import Tuple

import imageio
import numpy as np
# import torchvision
# from torch.utils.data import DataLoader
#
#datasetPath = "/pytorchtutorial/data/"
#
#
# def getDataLoaders(dataRoot: str, batchSize: int) -> Tuple[DataLoader, DataLoader]:
#     transforms = torchvision.transforms.Compose(
#             [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
#
#     trainDataset = torchvision.datasets.MNIST(
#             root=dataRoot,
#             train=True,
#             transform=transforms,
#             download=True,
#     )
#
#     valDataset = torchvision.datasets.MNIST(
#             root=dataRoot,
#             train=False,
#             transform=transforms,
#     )
#
#     trainDataLoader = DataLoader(
#             dataset=trainDataset,
#             batch_size=batchSize,
#             shuffle=True,
#     )
#
#     valDataLoader = DataLoader(
#             dataset=valDataset,
#             batch_size=batchSize,
#             shuffle=False,
#     )
#
#     return trainDataLoader, valDataLoader
#
#
def loadDataset():
    datasetPath = "data"
    datasetDict = {
        "train": {
            "images": loadIdx(f"{datasetPath}/train-images-idx3-ubyte"),
            "labels": loadIdx(f"{datasetPath}/train-labels-idx1-ubyte"),
        },
        "test": {
            "images": loadIdx(f"{datasetPath}/t10k-images-idx3-ubyte"),
            "labels": loadIdx(f"{datasetPath}/t10k-labels-idx1-ubyte"),
        },
    }
    return datasetDict


def loadIdx(idxFilePath):
    fileName = os.path.basename(idxFilePath)
    expression = ".*idx(\d)"
    matches = re.match(expression, fileName)
    if not matches:
        raise ValueError(f"File name did not match idx file convention: {fileName}")
    numChannels = int(matches.group(1))

    with open(idxFilePath, "rb") as f:
        buffer = f.read()
    data = parseData(buffer, numChannels)
    return data


def parseData(buffer, numChannels):
    headerBitSize = 32
    bitsPerByte = 8
    headerDataSize = headerBitSize//bitsPerByte
    magicNumberBase = 2048

    headerEnd = (numChannels+1)*headerDataSize
    header = buffer[:headerEnd]
    numberFormat = f">{'i'*(numChannels+1)}"
    parsedHeader = struct.unpack(numberFormat, header)
    magicNumber = parsedHeader[0]
    if magicNumber != magicNumberBase + numChannels:
        raise ValueError(f"Unexpected magic number: {magicNumber}")

    dataShape = parsedHeader[1:]
    data = np.frombuffer(buffer[headerEnd:], dtype=np.uint8)
    dataReshaped = data.reshape(dataShape)
    return dataReshaped


def main():
    trainImages = loadIdx(f"{datasetPath}/train-images-idx3-ubyte")
    trainLabels = loadIdx(f"{datasetPath}/train-labels-idx1-ubyte")
    for i, trainImage in enumerate(trainImages[:10]):
        outputPath = f"/tmp/output/image{i:02d}.png"
        imageio.imwrite(outputPath, trainImage)


if __name__ == "__main__":
    dataRoot = "/tmp/data"
    batchSize = 32
    _, _ = getDataLoaders(dataRoot, batchSize)

