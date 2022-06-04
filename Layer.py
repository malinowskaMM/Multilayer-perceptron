import numpy as np


class Layer:
    def __init__(self, inputNum, outputNum):
        self.input = None
        self.weights = np.random.randn(outputNum, inputNum)
        self.bias = np.random.randn(outputNum, 1)

    def loadWeights(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, outputDiff, alpha):
        weightsChange = np.dot(outputDiff, self.input.T)
        self.weights -= alpha * weightsChange
        self.bias -= alpha * outputDiff
        return np.dot(self.weights.T, outputDiff)