import unittest

import numpy as np

import mnist


class TestMnist(unittest.TestCase):
    def testcreateLabelsArray(self):
        # Given:
        N = 5
        y = np.arange(N, dtype=np.uint8)
        outputSize = 10
        expectedArray = np.zeros((N, outputSize))
        expectedArray[:5,:5] = np.eye(5)

        # When:
        computedArray = mnist.createLabelsArray(y, outputSize)

        # Then:
        self.assertTrue(np.allclose(expectedArray, computedArray))


if __name__ == "__main__":
    unittest.main()
