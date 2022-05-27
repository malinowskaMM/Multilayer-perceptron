import numpy as np


def _sigmoid(x, deriv=False):
    if deriv is True:
        return x * (1 - x)  # pochodna funkcji aktywacji
    return 1 / (1 + np.exp(-x))  # funkcja aktywacji - sigmoida


class Network:
    # inputNumber - ilosc neuronów wejściowych,
    # hiddenNumber - ilość neuronów w warstwie ukrytej
    # outputNumber - ilość neuronów wyjściowych
    # weightsIToH - wektor wag z warstwy wejściowej do warstwy ukrytej
    # weightsHToO - wektor wag z warstwy ukrytej do wartwy wyjściowej
    def __init__(self, inputNumber, hiddenNumber, outputNumber, weightsIToH, weightsHToO, biasChooser=False):
        self.counterBackward = 1
        self.ifBias = biasChooser
        self.inputNumber = inputNumber
        self.hiddenNumber = hiddenNumber
        self.outputNumber = outputNumber

        if weightsIToH is None:  # tablica input na hidden - każde wejście ma swoją wagę idąc do danego węzła
            self.weightsInputToHidden = np.random.rand(self.inputNumber, self.hiddenNumber)
        else:
            self.weightsInputToHidden = weightsIToH

        if weightsHToO is None:
            self.weightsHiddenToOutput = np.random.rand(self.hiddenNumber, self.outputNumber)
        else:
            self.weightsHiddenToOutput = weightsHToO

    def forwardPropagation(self, input):
        #signle row contains weights for one hidden neuron
        productOfWeightsAndInput = np.copy(self.weightsInputToHidden)
        for row in range(len(productOfWeightsAndInput)):
            productOfWeightsAndInput[row] *= input[row]

        #sum of hidden neuron is a sum of values in single column
        sumOfWeightsInputProduct = np.sum(productOfWeightsAndInput, axis=0)

        #for each sum we use sigmoid function
        sigmoidSumOfWeightsInputProduct = np.copy(sumOfWeightsInputProduct)
        for sum in range(len(sigmoidSumOfWeightsInputProduct)):
            sigmoidSumOfWeightsInputProduct[sum] = _sigmoid(sigmoidSumOfWeightsInputProduct[sum])

        ##signle row contains weights for one output neuron
        productOfWeightsAndHidden = np.copy(self.weightsHiddenToOutput)
        for element in range(len(productOfWeightsAndHidden)):
            productOfWeightsAndHidden[element] *= sigmoidSumOfWeightsInputProduct[element]

        # sum of output neuron is a sum of values in single column
        sumOfWeightsHiddenProduct = np.sum(productOfWeightsAndHidden, axis=0)

        # for each sum we use sigmoid function
        sigmoidSumOfWeightsHiddenProduct = np.copy(sumOfWeightsHiddenProduct)
        for sum in range(len(sigmoidSumOfWeightsHiddenProduct)):
            sigmoidSumOfWeightsHiddenProduct[sum] = _sigmoid(sigmoidSumOfWeightsHiddenProduct[sum])
        #print(self.weightsInputToHidden)
        #print(productOfWeightsAndInput)
        #print(sumOfWeightsInputProduct)
        #print(sigmoidSumOfWeightsInputProduct)
        #print(self.weightsHiddenToOutput)
        #print(productOfWeightsAndHidden)
        #print(sumOfWeightsHiddenProduct)
        #print(sigmoidSumOfWeightsHiddenProduct)



    # def backwardPropagation(self, input, expected, output):

    # def train(self, input, expected):
    # output = self.forwardPropagation(input)
    # self.backwardPropagation(input, expected, output)


o = Network(4, 2, 2, None, None)
table = np.array([2, 3, 4, 5])
o.forwardPropagation(table)
