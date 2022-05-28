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


def _castZerosOnesArrayToClassNames(array, threshold):
    if array == threshold * np.array([1, 0, 0]):
        return 'Iris-versicolor'
    elif array == threshold * np.array([0, 1, 0]):
        return 'Iris-setosa'
    elif array == threshold * np.array([0, 0, 1]):
        return 'Iris-virginica'


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

    def forwardPropagation(self, input, testingStats=None, testingMode=False, biasH=None, biasO=None):
        if biasH is None:
            biasH = np.zeros((self.inputNumber, self.hiddenNumber))
        if biasO is None:
            biasO = np.zeros((self.hiddenNumber, self.outputNumber))

        # signle row contains weights for one hidden neuron
        productOfWeightsAndInput = np.copy(self.weightsInputToHidden)
        for row in range(len(productOfWeightsAndInput)):
            productOfWeightsAndInput[row] *= input[row]
            productOfWeightsAndInput[row] += biasH[row]

        # sum of hidden neuron is a sum of values in single column - net h
        self.sumOfWeightsInputProduct = np.sum(productOfWeightsAndInput, axis=0)

        # for each sum we use sigmoid function - out h
        self.sigmoidSumOfWeightsInputProduct = _sigmoid(self.sumOfWeightsInputProduct)

        # signle row contains weights for one output neuron
        productOfWeightsAndHidden = np.copy(self.weightsHiddenToOutput)
        for elementIter in range(len(productOfWeightsAndHidden)):
            productOfWeightsAndHidden[elementIter] *= self.sigmoidSumOfWeightsInputProduct[elementIter]
            productOfWeightsAndHidden[elementIter] += biasO[elementIter]

        # sum of output neuron is a sum of values in single column - net o
        sumOfWeightsHiddenProduct = np.sum(productOfWeightsAndHidden, axis=0)

        # for each sum we use sigmoid function - out o
        sigmoidSumOfWeightsHiddenProduct = _sigmoid(sumOfWeightsHiddenProduct)

        if testingMode:
            testingStats.append(input)  # wzorzec wejsciowy
            testingStats.append(sigmoidSumOfWeightsHiddenProduct)  # wartosc na wyjsciu
            testingStats.append(self.weightsHiddenToOutput)  # wagi neuronow wyjsciowych
            testingStats.append(self.sigmoidSumOfWeightsInputProduct)  # wartosci wyjsciowych neuronów ukrytych
            testingStats.append(self.weightsInputToHidden)  # wagi neuronow ukrytych

        return sigmoidSumOfWeightsHiddenProduct

    # alpha - step length coefficient
    def backwardPropagation(self, input, expected, outputForward, epochNumber, errorsOfEpoch, alpha=0.6):
        # calculate sum of errors (global error) between expected and output
        errorSum = _sum(outputForward, expected)  # Cost function
        if epochNumber % 10 == 0:
            errorsOfEpoch.append(errorSum)

        # weights hidden to output modifications
        modifyHiddenToOutputWeights = np.copy(self.weightsHiddenToOutput)
        deltaModifyHiddenToOutputWeights = np.copy(self.weightsHiddenToOutput)
        for weightRowIter in range(len(modifyHiddenToOutputWeights)):
            for weightColIter in range(len(modifyHiddenToOutputWeights[0])):
                deltaModifyHiddenToOutputWeights[weightRowIter][weightColIter] *= _sum(outputForward[weightColIter], expected[weightColIter], True)
                deltaModifyHiddenToOutputWeights[weightRowIter][weightColIter] *= _sigmoid(outputForward[weightColIter], True)
                modifyHiddenToOutputWeights[weightRowIter][weightColIter] = deltaModifyHiddenToOutputWeights[weightRowIter][weightColIter]
                modifyHiddenToOutputWeights[weightRowIter][weightColIter] *= self.sigmoidSumOfWeightsInputProduct[weightRowIter]

        sumDeltaModifyHiddenToOutputWeights = np.sum(deltaModifyHiddenToOutputWeights, axis = 1)
        self.weightsHiddenToOutput -= modifyHiddenToOutputWeights * alpha

        # sum of output neuron is a sum of values in single column
        sumOfWeightsHiddenProduct = np.sum(np.copy(self.weightsHiddenToOutput), axis=1)

        # weights input to hidden modifications
        modifyInputToHiddenWeights = np.copy(self.weightsInputToHidden)
        for weightRowIter in range(len(modifyInputToHiddenWeights)):
            for weightColIter in range(len(modifyInputToHiddenWeights[0])):
                modifyInputToHiddenWeights[weightRowIter][weightColIter] = _sigmoid(self.sigmoidSumOfWeightsInputProduct[weightColIter], True)
                modifyInputToHiddenWeights[weightRowIter][weightColIter] *= sumDeltaModifyHiddenToOutputWeights[weightColIter]
                modifyInputToHiddenWeights[weightRowIter][weightColIter] *= input[weightRowIter]

        self.weightsInputToHidden -= modifyInputToHiddenWeights * alpha

    def trainNew(self, input, expected, epochNum):
        errorsOfEpoch = []
        for i in range(epochNum):
            for j in range(len(input)):
                output = self.forwardPropagation(input[j])
                self.backwardPropagation(input[j], _castClassNamesToZerosOnesArray(expected[j]), output, epochNum, errorsOfEpoch)
        return errorsOfEpoch

    def testingOne(self, input, expected):
        # 1. wzorzec wejsciowy
        # 2. wartosc na wyjsciu
        # 3. wagi neuronow wyjsciowych
        # 4. wartosci wyjsciowych neuronów ukrytych
        # 5. wagi neuronow ukrytych
        testingStats = []
        output = self.forwardPropagation(input, testingStats, True)
        testingStats.append(expected)  # 6. pożadany wzorzec odpowiedzi
        testingStats.append(expected-output)  # 7. bład na poszeczgólnych wyjsciach sieci
        testingStats.append(_sum(output, expected))  # 8. popełniony przez siec bład dla całego wzorca
        return testingStats
