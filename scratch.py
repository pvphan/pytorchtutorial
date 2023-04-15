import numpy as np

import mnistdataset


def relu(x: np.ndarray):
    # returns the activation of the unit
    return x * (x > 0)


def forward(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return inputs @ weights


def main():
    dataset = mnistdataset.loadDataset()


if __name__ == "__main__":
    main()
