import pprint
from typing import List, Tuple

import numpy as np


class FullyConnectedNet:
    _learningRate = 0.05
    def __init__(self, weights: List[List[List[float]]]):
        self._weights = weights
        self._nodeValues = [[] for _ in range(len(self._weights) + 1)]

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
        hiddenLayer = inputs
        for i, weightsLayer in enumerate(self._weights):
            self._nodeValues[i] = hiddenLayer
            hiddenLayer = forwardOp(hiddenLayer, weightsLayer)
        self._nodeValues[-1] = hiddenLayer
        print(f"_nodeValues")
        pprint.pprint(self._nodeValues)
        return hiddenLayer

    def backprop(self, prediction: List[float], labels: List[float]) -> None:
        Δ = np.array(prediction) - np.array(labels)
        λ = self._learningRate
        for i in range(len(self._weights) - 1, -1, -1):
            initialWeightsLayer = np.array(self._weights[i])
            print(f"initialWeightsLayer{i}")
            pprint.pprint(initialWeightsLayer)
            hiddenLayer = self._nodeValues[i]
            print(f"hiddenLayer{i}")
            pprint.pprint(hiddenLayer)
            updatedWeightsLayer = initialWeightsLayer - (λ * Δ * np.array(hiddenLayer)).reshape(-1, initialWeightsLayer.shape[1])
            print(f"updatedWeightsLayer{i}")
            pprint.pprint(updatedWeightsLayer)
            self._weights[i] = updatedWeightsLayer.tolist()
            print()


def relu(x: np.ndarray):
    # returns the activation of the unit
    return x * (x > 0)


def forwardOp(inputs: List[float], weightsLayer: List[List[float]]) -> List[float]:
    output = (np.array(inputs) @ np.array(weightsLayer)).tolist()
    return output


def computeError(outputs: List[float], label: List[float]) -> float:
    return (0.5 * (np.array(label) - np.array(outputs)) ** 2).tolist()
