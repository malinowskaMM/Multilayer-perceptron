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
    def __init__(self, inputNumber, hiddenNumber, outputNumber, momentum=0):
        self.counterBackward = 1
        self.inputNumber = inputNumber
        self.hiddenNumber = hiddenNumber
        self.outputNumber = outputNumber
        self.weightsInputToHidden = np.random.rand(self.inputNumber, self.hiddenNumber)
        self.weightsHiddenToOutput = np.random.rand(self.hiddenNumber, self.outputNumber)
        self.momentum = momentum
        self.weightIToHIncrement = np.zeros((self.inputNumber, self.hiddenNumber))
        self.weightHToOIncrement = np.zeros((self.hiddenNumber, self.outputNumber))

    def forwardPropagation(self, input, biasH=0, biasO=0, testingStats=None, testingMode=False):
        #single row contains weights for one hidden neuron
        productOfWeightsAndInput = np.copy(self.weightsInputToHidden)
        for row in range(len(productOfWeightsAndInput)):
            productOfWeightsAndInput[row] *= input[row]
            productOfWeightsAndInput[row] += biasH

        # sum of hidden neuron is a sum of values in single column - net h
        self.sumOfWeightsInputProduct = np.sum(productOfWeightsAndInput, axis=0)

        # for each sum we use sigmoid function - out h
        self.sigmoidSumOfWeightsInputProduct = _sigmoid(self.sumOfWeightsInputProduct)

        #single row contains weights for one output neuron
        productOfWeightsAndHidden = np.copy(self.weightsHiddenToOutput)
        for elementIter in range(len(productOfWeightsAndHidden)):
            productOfWeightsAndHidden[elementIter] *= self.sigmoidSumOfWeightsInputProduct[elementIter]
            productOfWeightsAndHidden[elementIter] += biasO
        # sum of output neuron is a sum of values in single column
        self.sumOfWeightsHiddenProduct = np.sum(productOfWeightsAndHidden, axis=0)  # vector of inputs into output neurons

        # for each sum we use sigmoid function - out o
        sigmoidSumOfWeightsHiddenProduct = _sigmoid(self.sumOfWeightsHiddenProduct)

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
            if errorsOfEpoch is not None:
                errorsOfEpoch.append(errorSum)

        # weights hidden to output modifications
        weightsHtOBeforeChange = np.copy(self.weightsHiddenToOutput)  # needed for next layer calculations
        modifyHiddenToOutputWeights = np.zeros((self.hiddenNumber, self.outputNumber))
        # modifyHiddenToOutputWeights :  rows = hidden, col = output
        for row in range(self.hiddenNumber):
            for col in range(self.outputNumber):
                change = -1 * (expected[col] - outputForward[col]) * outputForward[col] * (1 - outputForward[col]) * self.weightsHiddenToOutput[row][col]
                temp = change
                change += self.momentum * self.weightHToOIncrement[row][col]
                self.weightsHiddenToOutput[row][col] -= alpha * change  # weight modification (hidden to output)
                self.weightHToOIncrement[row][col] = temp  # update weight change matrix


        # weights input to hidden modifications
        modifyInputToHiddenWeights = np.copy(self.weightsInputToHidden)
        # modifyInputToHiddenWeights :  rows = input, col = hidden

        # weights used here are weightsHtOBeforeChange, NOT self.weightsHiddenToOutput (which is modified at this point)
        for row in range(self.inputNumber):
            for col in range(self.hiddenNumber):
                change = 0
                for outputIter in range(self.outputNumber):
                    x = -1 * (expected[outputIter] - outputForward[outputIter]) * outputForward[outputIter] * (
                                1 - outputForward[outputIter]) * self.weightsInputToHidden[row][col]
                    y = self.sumOfWeightsInputProduct[col] * (1 - self.sumOfWeightsInputProduct[col])
                    z = input[row]
                    change = x * y * z  # dla irysów szybko rośnie poza zakres, dla autoencoder już nie
                    temp = change
                    change += self.momentum * self.weightIToHIncrement[row][col]
                    self.weightsInputToHidden[row][col] -= alpha * change
                    self.weightIToHIncrement[row][col] = temp  # update weight change matrix

    def backwardPropagationOld(self, input, expected, outputForward, epochNumber, errorsOfEpoch, alpha=0.6):
        # calculate sum of errors (global error) between expected and output
        errorSum = _sum(outputForward, expected)  # Cost function
        if epochNumber % 10 == 0:
            if errorsOfEpoch is not None:
                errorsOfEpoch.append(errorSum)
        # weights hidden to output modifications
        modifyHiddenToOutputWeights = np.copy(self.weightsHiddenToOutput)
        deltaModifyHiddenToOutputWeights = np.copy(self.weightsHiddenToOutput)
        for weightRowIter in range(len(modifyHiddenToOutputWeights)):
            for weightColIter in range(len(modifyHiddenToOutputWeights[0])):
                deltaModifyHiddenToOutputWeights[weightRowIter][weightColIter] *= _sum(outputForward[weightColIter],
                                                                                       expected[weightColIter],
                                                                                       True)
                deltaModifyHiddenToOutputWeights[weightRowIter][weightColIter] *= _sigmoid(
                    outputForward[weightColIter], True)
                modifyHiddenToOutputWeights[weightRowIter][weightColIter] = \
                deltaModifyHiddenToOutputWeights[weightRowIter][weightColIter]
                modifyHiddenToOutputWeights[weightRowIter][weightColIter] *= self.sigmoidSumOfWeightsInputProduct[
                    weightRowIter]
        sumDeltaModifyHiddenToOutputWeights = np.sum(deltaModifyHiddenToOutputWeights, axis=1)
        self.weightsHiddenToOutput -= modifyHiddenToOutputWeights * alpha
        # sum of output neuron is a sum of values in single column
        sumOfWeightsHiddenProduct = np.sum(np.copy(self.weightsHiddenToOutput), axis=1)
        # weights input to hidden modifications
        modifyInputToHiddenWeights = np.copy(self.weightsInputToHidden)
        for weightRowIter in range(len(modifyInputToHiddenWeights)):
            for weightColIter in range(len(modifyInputToHiddenWeights[0])):
                modifyInputToHiddenWeights[weightRowIter][weightColIter] = _sigmoid(
                    self.sigmoidSumOfWeightsInputProduct[weightColIter], True)
                modifyInputToHiddenWeights[weightRowIter][weightColIter] *= sumDeltaModifyHiddenToOutputWeights[
                    weightColIter]
                modifyInputToHiddenWeights[weightRowIter][weightColIter] *= input[weightRowIter]
        self.weightsInputToHidden -= modifyInputToHiddenWeights * alpha

    def train(self, input, expected):
        output = self.forwardPropagation(input)
        # self.backwardPropagation(input, expected, output)

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
        output = self.forwardPropagation(input, 0, 0, testingStats, True)
        testingStats.append(expected)  # 6. pożadany wzorzec odpowiedzi
        testingStats.append(expected-output)  # 7. bład na poszeczgólnych wyjsciach sieci
        testingStats.append(_sum(output, expected))  # 8. popełniony przez siec bład dla całego wzorca
        return testingStats
