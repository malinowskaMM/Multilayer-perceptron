import numpy as np

def _sigmoid(x, deriv=False):
    if deriv is True:
        return np.exp(x)/((np.exp(x) + 1)**2) # pochodna funkcji aktywacji
    return 1 / (1 + np.exp(-x)) # funkcja aktywacji - sigmoida

class Network:
    # inputNumber - ilosc neuronów wejściowych,
    # hiddenNumber - ilość neuronów w warstwie ukrytej
    # outputNumber - ilość neuronów wyjściowych
    # weightsIToH - wektor wag z warstwy wejściowej do warstwy ukrytej
    # weightsHToO - wektor wag z warstwy ukrytej do wartwy wyjściowej
    def __init__(self, inputNumber, hiddenNumber, outputNumber, weightsIToH, weightsHToO, bias=False):
        self. counterBackward = 1
        self.ifBias = bias
        self.z2Delta = None
        self.z2Error = None
        self.outputDelta = None
        self.outputError = None
        self.z3 = None
        self.z2 = None
        self.z = None
        self.inputNumber = inputNumber
        self.hiddenNumber = hiddenNumber
        self.outputNumber = outputNumber

        if weightsIToH is None:
            self.weightsInputToHidden = np.random.rand(self.inputNumber, self.hiddenNumber)
        else:
            self.weightsInputToHidden = weightsIToH

        if weightsHToO is None:
            self.weightsHiddenToOutput = np.random.rand(self.inputNumber, self.hiddenNumber)
        else:
            self.weightsHiddenToOutput = weightsHToO

    def forwardPropagation(self, input):
        bias = 0
        if self.ifBias is True:
            bias = 1
        self.z = np.dot(input, self.weightsInputToHidden) + bias  # dot product of input and first set of weights
        self.z2 = _sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.weightsHiddenToOutput) + bias  # dot product of hidden layer and second set of weights
        output = _sigmoid(self.z3)
        return output

    # from wikipedia

    def backwardPropagation(self, input, expected, output):
        self.outputError = expected - output
        self.outputDelta = self.outputError * _sigmoid(output, deriv=True)

        self.z2Error = self.outputDelta.dot(self.weightsHiddenToOutput.T)
        self.z2Delta = self.z2Error * _sigmoid(self.z2, deriv=True)

        self.weightsInputToHidden += self.counterBackward * input.T.dot(self.z2Delta)
        self.weightsHiddenToOutput += self.counterBackward * self.z2.T.dot(self.outputDelta)

        self.counterBackward += 1

    def train(self, input, expected):
        output = self.forwardPropagation(input)
        self.backwardPropagation(input, expected, output)

