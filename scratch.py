from typing import List

import numpy as np

import mnistdataset


class FullyConnectedNet:
    def __init__(self, layerSizes: List[int]):
        self._layerSizes = layerSizes
        self._weights = [np.random.uniform(
                low=-1, high=+1, size=(self._layerSizes[i], self._layerSizes[i+1]))
                         for i in range(len(self._layerSizes) - 1)]

    def forward(self):
        inputs = np.ones(self._layerSizes[0], dtype=np.float32)
        outputs = inputs
        for inputLayerIndex in range(1, len(self._layerSizes)):
            outputs = forward(inputs, self._weights[inputLayerIndex-1])
            inputs = outputs

        return outputs


def relu(x: np.ndarray):
    # returns the activation of the unit
    return x * (x > 0)


def forward(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return inputs @ weights


def main():
    dataset = mnistdataset.loadDataset()


if __name__ == "__main__":
    main()
