import pprint
from typing import List, Tuple

import numpy as np


class FullyConnectedNet:
    def __init__(self, weights: List[List[List[float]]]):
        self._weights = weights

    def __repr__(self) -> str:
        reprString = pprint.pformat(self._weights)
        return reprString

    @classmethod
    def fromRandomWeights(cls, layerSizes: List[int]):
        weights = []
        for i in range(len(layerSizes) - 1):
            weightsLayer = []
            for _ in range(layerSizes[i]):
                nodeWeights = [np.random.uniform(low=-1, high=+1) for _ in range(layerSizes[i+1])]
                weightsLayer.append(nodeWeights)
            weights.append(weightsLayer)
        return cls(weights)

    def forward(self, inputs: List[float]) -> List[float]:
        outputs = inputs
        for weightsLayer in self._weights:
            outputs = forwardOp(outputs, weightsLayer)
        return outputs


def relu(x: np.ndarray):
    # returns the activation of the unit
    return x * (x > 0)


def forwardOp(inputs: List[float], weightsLayer: List[List[float]]) -> List[float]:
    output = (np.array(inputs) @ np.array(weightsLayer)).tolist()
    return output


def computeError(outputs: List[float], label: List[float]) -> float:
    return (0.5 * (np.array(label) - np.array(outputs)) ** 2).tolist()
