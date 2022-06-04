import numpy as np
import Layer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Sigmoid:
    def __init__(self):
        self.a = 1

    def forward(self, input):
        return sigmoid(input)

    def backward(self, outputDiff, alpha):
        deriv = sigmoid(outputDiff) * (1 - sigmoid(outputDiff))
        return np.multiply(outputDiff, deriv)
