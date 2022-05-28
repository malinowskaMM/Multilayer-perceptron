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
    return errorSum / 2

def _castClassNamesToZerosOnesArray(className):
    if className == 'Iris-versicolor':
        return [1, 0, 0]
    elif className == 'Iris-setosa':
        return [0, 1, 0]
    elif className == 'Iris-virginica':
        return [0, 0, 1]

class Network:
    # inputNumber - ilosc neuronów wejściowych,
    # hiddenNumber - ilość neuronów w warstwie ukrytej
    # outputNumber - ilość neuronów wyjściowych
    # weightsIToH - wektor wag z warstwy wejściowej do warstwy ukrytej
    # weightsHToO - wektor wag z warstwy ukrytej do wartwy wyjściowej
    def __init__(self, inputNumber, hiddenNumber, outputNumber, weightsIToH, weightsHToO, momentum, biasChooser=False):
        self.counterBackward = 1
        self.ifBias = biasChooser
        self.inputNumber = inputNumber
        self.hiddenNumber = hiddenNumber
        self.outputNumber = outputNumber
        self.momentum = momentum
        self.weightIToHIncrement = np.zeros((self.inputNumber, self.hiddenNumber))
        self.weightHToOIncrement = np.zeros((self.hiddenNumber, self.outputNumber))

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
        productOfWeightsAndInput = np.copy(self.weightsInputToHidden)
        for row in range(len(productOfWeightsAndInput)):
            productOfWeightsAndInput[row] *= input[row]
            productOfWeightsAndInput[row] += biasH[row]

        #sum of hidden neuron is a sum of values in single column - net h
        self.sumOfWeightsInputProduct = np.sum(productOfWeightsAndInput, axis=0)

        #for each sum we use sigmoid function - out h
        self.sigmoidSumOfWeightsInputProduct = _sigmoid(self.sumOfWeightsInputProduct)

        ##signle row contains weights for one output neuron
        productOfWeightsAndHidden = np.copy(self.weightsHiddenToOutput)
        for elementIter in range(len(productOfWeightsAndHidden)):
            productOfWeightsAndHidden[elementIter] *= self.sigmoidSumOfWeightsInputProduct[elementIter]
            productOfWeightsAndHidden[elementIter] += biasO[elementIter]

        # sum of output neuron is a sum of values in single column - net o
        sumOfWeightsHiddenProduct = np.sum(productOfWeightsAndHidden, axis=0)


        # for each sum we use sigmoid function - out o
        sigmoidSumOfWeightsHiddenProduct = _sigmoid(sumOfWeightsHiddenProduct)

        # print(self.weightsInputToHidden)
        # print(productOfWeightsAndInput)
        # print(sumOfWeightsInputProduct)
        # print(self.sigmoidSumOfWeightsInputProduct)
        # print(self.weightsHiddenToOutput)
        # print(productOfWeightsAndHidden)
        # print(sumOfWeightsHiddenProduct)
        # print(sigmoidSumOfWeightsHiddenProduct)
        return sigmoidSumOfWeightsHiddenProduct



    #alpha - step length coefficient
    def backwardPropagation(self, input, expected, outputForward, alpha = 0.6):
        #calculate sum of errors (global error) between expected and output
        errorSum = _sum(outputForward, expected)  #Cost function

        #weights hidden to output modifications
        modifyHiddenToOutputWeights = np.copy(self.weightsHiddenToOutput)
        deltaModifyHiddenToOutputWeights = np.copy(self.weightsHiddenToOutput)
        for rowIt in range(len(modifyHiddenToOutputWeights)):
            for colIt in range(len(modifyHiddenToOutputWeights[0])):
                deltaModifyHiddenToOutputWeights[rowIt][colIt] *= _sum(outputForward[colIt], expected[colIt], True)
                deltaModifyHiddenToOutputWeights[rowIt][colIt] *= _sigmoid(outputForward[colIt], True)
                modifyHiddenToOutputWeights[rowIt][colIt] = deltaModifyHiddenToOutputWeights[rowIt][colIt]
                modifyHiddenToOutputWeights[rowIt][colIt] *= self.sigmoidSumOfWeightsInputProduct[rowIt]
                tempModification = np.copy(modifyHiddenToOutputWeights[rowIt][colIt])
                modifyHiddenToOutputWeights[rowIt][colIt] += self.momentum * self.weightHToOIncrement[rowIt][colIt]
                self.weightHToOIncrement[rowIt][colIt] = tempModification

        sumDeltaModifyHiddenToOutputWeights = np.sum(deltaModifyHiddenToOutputWeights, axis = 1)
        self.weightsHiddenToOutput -= modifyHiddenToOutputWeights * alpha

        # sum of output neuron is a sum of values in single column
        sumOfWeightsHiddenProduct = np.sum(np.copy(self.weightsHiddenToOutput), axis=1)

        #weights input to hidden modifications
        modifyInputToHiddenWeights = np.copy(self.weightsInputToHidden)
        for rowIt in range(len(modifyInputToHiddenWeights)):
            for colIt in range(len(modifyInputToHiddenWeights[0])):
                modifyInputToHiddenWeights[rowIt][colIt] = _sigmoid(self.sigmoidSumOfWeightsInputProduct[colIt], True)
                modifyInputToHiddenWeights[rowIt][colIt] *= sumDeltaModifyHiddenToOutputWeights[colIt]
                modifyInputToHiddenWeights[rowIt][colIt] *= input[rowIt]
                tempModification = np.copy(modifyInputToHiddenWeights[rowIt][colIt])
                modifyInputToHiddenWeights[rowIt][colIt] += self.momentum * self.weightIToHIncrement[rowIt][colIt]
                self.weightIToHIncrement[rowIt][colIt] = tempModification


        self.weightsInputToHidden -= modifyInputToHiddenWeights * alpha
        #print(modifyInputToHiddenWeights)



    def trainNew(self, input, expected, epochNum):
        for i in range(epochNum):
            for j in range(len(input)):
                output = self.forwardPropagation(input[j])
                globalError = _sum(output, _castClassNamesToZerosOnesArray(expected[j]))
                print(globalError)
                self.backwardPropagation(input[j],  _castClassNamesToZerosOnesArray(expected[j]), output)

    def trainTest(self, input, expected, epochNum):
        for i in range(epochNum):
            for j in range(len(input)):
                output = self.forwardPropagation(input[j])
                globalError = _sum(output, expected[j])
                print(globalError)
                self.backwardPropagation(input[j],  expected[j], output)