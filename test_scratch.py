import unittest

import numpy as np

import scratch


class TestFullyConnectedNet(unittest.TestCase):
    def setUp(self):
        # From example: https://hmkcode.com/ai/backpropagation-step-by-step/
        layer1Weights = [
            [0.11, 0.12],
            [0.21, 0.08],
        ]
        layer2Weights = [
            [0.14],
            [0.15],
        ]
        self.weights = [
                layer1Weights,
                layer2Weights,
        ]
        self.inputs = [2.0, 3.0]

    def testforward(self):
        # Given:
        layerSizes = [5, 3, 3]
        network = scratch.FullyConnectedNet.fromRandomWeights(layerSizes)
        inputs = [1.0 for _ in range(layerSizes[0])]

        # When:
        outputs = network.forward(inputs)

        # Then:
        self.assertEqual(len(outputs), layerSizes[-1])

    def testforwardFromWeights(self):
        # Given:
        weights = self.weights
        inputs = self.inputs
        network = scratch.FullyConnectedNet(weights)

        # When:
        outputs = network.forward(inputs)

        # Then:
        self.assertEqual(len(outputs), len(weights[-1][-1]))
        self.assertAlmostEqual(outputs[0], 0.191)

    def testbackprop(self):
        # Given:
        weights = self.weights
        inputs = self.inputs
        network = scratch.FullyConnectedNet(weights)
        labels = [1.0]
        numIters = 100

        # When:
        for _ in range(numIters):
            outputs = network.forward(inputs)
            network.backprop(outputs, labels)

        # Then:
        self.assertAlmostEqual(outputs[0], labels[0])


class TestScratch(unittest.TestCase):
    def testrelu(self):
        # Given:
        x = np.random.uniform(low=-1, high=+1, size=(9,))

        # When:
        g = scratch.relu(x)

        # Then:
        self.assertEqual(x.shape, g.shape)
        for i in range(x.shape[0]):
            if x[i] < 0:
                self.assertAlmostEqual(g[i], 0)
            else:
                self.assertAlmostEqual(g[i], x[i])

    def testforwardOp(self):
        # Given:
        a1 = 1
        a2 = 1
        inputLayer = np.array([a1, a2])
        outputLayerSize = 3
        weights = np.random.uniform(
                low=-1, high=+1, size=(len(inputLayer), outputLayerSize))

        # When:
        output = scratch.forwardOp(inputLayer, weights)

        # Then:
        self.assertEqual(len(output), outputLayerSize)

    def testcomputeError(self):
        # Given:
        outputs = [0.191]
        labels = [1.0]

        # When:
        error = scratch.computeError(outputs, labels)

        # Then:
        self.assertAlmostEqual(error[0], 0.3272405)


if __name__ == "__main__":
    unittest.main()
