import unittest

import numpy as np

import scratch


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

    def testforward(self):
        # Given:
        a1 = 1
        a2 = 1
        inputLayer = np.array([a1, a2])
        outputLayerSize = 3
        weights = np.random.uniform(
                low=-1, high=+1, size=(len(inputLayer), outputLayerSize))

        # When:
        output = scratch.forward(inputLayer, weights)

        # Then:
        self.assertEqual(output.shape, (outputLayerSize,))


if __name__ == "__main__":
    unittest.main()
