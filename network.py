import numpy as np


def _sigmoid(x, deriv=False):
    if deriv is True:
        return x * (1 - x)  # pochodna funkcji aktywacji
    return 1 / (1 + np.exp(-x))  # funkcja aktywacji - sigmoida

def _sum(outputForward, expected ,deriv=False):
    errorSum = 0
    if deriv is True: #outputForward and expected as number not array
        return (expected - outputForward) * (-1)
    for i in range(len(outputForward)):
        errorSum += (expected[i] - outputForward[i]) ** 2
        errorSum /= 2
    return errorSum

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

    def forwardPropagation(self, input, biasH=None, biasO=None):
        if biasH is None:
            biasH = np.zeros((self.inputNumber, self.hiddenNumber))
        if biasO is None:
            biasO = np.zeros((self.hiddenNumber, self.outputNumber))

        #signle row contains weights for one hidden neuron
        self.productOfWeightsAndInput = np.copy(self.weightsInputToHidden)
        for row in range(len(self.productOfWeightsAndInput)):
            self.productOfWeightsAndInput[row] *= input[row]
            self.productOfWeightsAndInput[row] += biasH[row]

        #sum of hidden neuron is a sum of values in single column
        sumOfWeightsInputProduct = np.sum(self.productOfWeightsAndInput, axis=0)

        #for each sum we use sigmoid function
        self.sigmoidSumOfWeightsInputProduct = np.copy(sumOfWeightsInputProduct)
        for sumIter in range(len(self.sigmoidSumOfWeightsInputProduct)):
            self.sigmoidSumOfWeightsInputProduct[sumIter] = _sigmoid(self.sigmoidSumOfWeightsInputProduct[sumIter])
        #print(sigmoidSumOfWeightsInputProduct)

        ##signle row contains weights for one output neuron
        self.productOfWeightsAndHidden = np.copy(self.weightsHiddenToOutput)
        for elementIter in range(len(self.productOfWeightsAndHidden)):
            self.productOfWeightsAndHidden[elementIter] *= self.sigmoidSumOfWeightsInputProduct[elementIter]
            self.productOfWeightsAndHidden[elementIter] += biasO[elementIter]

        # sum of output neuron is a sum of values in single column
        sumOfWeightsHiddenProduct = np.sum(self.productOfWeightsAndHidden, axis=0)


        # for each sum we use sigmoid function
        sigmoidSumOfWeightsHiddenProduct = np.copy(sumOfWeightsHiddenProduct)
        for sumIter in range(len(sigmoidSumOfWeightsHiddenProduct)):
            sigmoidSumOfWeightsHiddenProduct[sumIter] = _sigmoid(sigmoidSumOfWeightsHiddenProduct[sumIter])

        # print(self.weightsInputToHidden)
        # print(productOfWeightsAndInput)
        # print(sumOfWeightsInputProduct)
        # print(self.sigmoidSumOfWeightsInputProduct)
        # print(self.weightsHiddenToOutput)
        # print(productOfWeightsAndHidden)
        # print(sumOfWeightsHiddenProduct)
        # print(sigmoidSumOfWeightsHiddenProduct)
        output = sigmoidSumOfWeightsHiddenProduct
        return output



    #alpha - step length coefficient
    def backwardPropagation(self, input, expected, outputForward, alpha = 1):
        #calculate sum of errors (global error) between expected and output
        errorSum = _sum(outputForward, expected)  #Cost function
        divErrorSum = 0
        for i in range(len(outputForward)):
            divErrorSum += _sum(outputForward[i], expected[i], True)
        # print(divErrorSum)

        #weights hidden to output modifications
        modifyHiddenToOutputWeights = np.copy(self.weightsHiddenToOutput)
        for weightRowIter in range(len(modifyHiddenToOutputWeights)):
            for weightColIter in range(len(modifyHiddenToOutputWeights[0])):
                modifyHiddenToOutputWeights[weightRowIter][weightColIter] *= _sum(outputForward[weightColIter], expected[weightColIter], True)
                modifyHiddenToOutputWeights[weightRowIter][weightColIter] *= _sigmoid(self.productOfWeightsAndHidden[weightColIter], True)


        print((self.weightsHiddenToOutput))
        self.weightsHiddenToOutput -= modifyHiddenToOutputWeights
        print((self.weightsHiddenToOutput))

        # sum of output neuron is a sum of values in single column
        sumOfWeightsHiddenProduct = np.sum(np.copy(self.weightsHiddenToOutput), axis=1)
        print(sumOfWeightsHiddenProduct)

        #weights input to hidden modifications
        modifyInputToHiddenWeights = np.copy(self.weightsInputToHidden)
        for weightRowIter in range(len(modifyInputToHiddenWeights)):
            for weightColIter in range(len(modifyInputToHiddenWeights[0])):
                modifyInputToHiddenWeights[weightRowIter][weightColIter] = _sigmoid( self.productOfWeightsAndInput[weightColIter], True)
                modifyInputToHiddenWeights[weightRowIter][weightColIter] *= sumOfWeightsHiddenProduct[weightColIter % 2]
                modifyInputToHiddenWeights[weightRowIter][weightColIter] *= input[weightRowIter][weightColIter]


        self.weightsInputToHidden -= modifyInputToHiddenWeights
        #print(modifyInputToHiddenWeights)


    def train(self, input, expected):
        output = self.forwardPropagation(input)
        # self.backwardPropagation(input, expected, output)



