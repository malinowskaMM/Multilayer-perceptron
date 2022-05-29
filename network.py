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
    def __init__(self, inputNumber, hiddenNumber, outputNumber, weightsIToH=None, weightsHToO=None, momentum=0):
        self.counterBackward = 1
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
        self.momentum = momentum
        self.weightIToHIncrement = np.zeros((self.inputNumber, self.hiddenNumber))
        self.weightHToOIncrement = np.zeros((self.hiddenNumber, self.outputNumber))

    def forwardPropagation(self, input, biasH=0, biasO=0, testingStats=None, testingMode=False):
        # single row contains weights for one hidden neuron

        productOfWeightsAndInput = np.copy(self.weightsInputToHidden)
        for row in range(len(productOfWeightsAndInput)):
            productOfWeightsAndInput[row] *= input[row]
            # productOfWeightsAndInput[row] += biasH

        # sum of hidden neuron is a sum of values in single column - net h
        self.sumOfWeightsInputProduct = np.sum(productOfWeightsAndInput, axis=0)

        # for each sum we use sigmoid function - out h
        self.sigmoidSumOfWeightsInputProduct = _sigmoid(self.sumOfWeightsInputProduct)

        # single row contains weights for one output neuron
        productOfWeightsAndHidden = np.copy(self.weightsHiddenToOutput)
        for elementIter in range(len(productOfWeightsAndHidden)):
            productOfWeightsAndHidden[elementIter] *= self.sigmoidSumOfWeightsInputProduct[elementIter]
            # productOfWeightsAndHidden[elementIter] += biasO
        # sum of output neuron is a sum of values in single column
        self.sumOfWeightsHiddenProduct = np.sum(productOfWeightsAndHidden,
                                                axis=0)  # vector of inputs into output neurons

        # for each sum we use sigmoid function - out o
        sigmoidSumOfWeightsHiddenProduct = _sigmoid(self.sumOfWeightsHiddenProduct)

        if testingMode:
            testingStats.append(input)  # wzorzec wejsciowy
            testingStats.append(sigmoidSumOfWeightsHiddenProduct)  # wartosc na wyjsciu
            testingStats.append(self.weightsHiddenToOutput)  # wagi neuronow wyjsciowych
            testingStats.append(self.sigmoidSumOfWeightsInputProduct)  # wartosci wyjsciowych neuronów ukrytych
            testingStats.append(self.weightsInputToHidden)  # wagi neuronow ukrytych

        return sigmoidSumOfWeightsHiddenProduct


    def backwardPropagationOld(self, input, expected, outputForward, epochNumber, errorsOfEpoch, alpha=1):
        # calculate sum of errors (global error) between expected and output
        # errorSum = _sum(outputForward, expected)  # Cost function
        # if epochNumber % 10 == 0:
        #     if errorsOfEpoch is not None:
        #         errorsOfEpoch.append(errorSum)

        errorsArray = expected - outputForward
        for i in range(len(errorsArray)):
            errorsArray[i] *= _sigmoid(errorsArray[i], True)

        weightsHiddenToOutputModification = np.copy(self.weightsHiddenToOutput)
        for row in range(len(weightsHiddenToOutputModification)):
            for col in range(len(weightsHiddenToOutputModification[0])):
                weightsHiddenToOutputModification[row][col] *= errorsArray[row]
        # for row in range(len(weightsHiddenToOutputModification)):
        #         weightsHiddenToOutputModification[row] *= errorsArray
        print(weightsHiddenToOutputModification)


        modifyWeightsHiddenToOutput = np.copy(self.weightsHiddenToOutput)
        print(modifyWeightsHiddenToOutput)
        for row in range(len(modifyWeightsHiddenToOutput)):
            for col in range(len(modifyWeightsHiddenToOutput[0])):
                modifyWeightsHiddenToOutput[row][col] += modifyWeightsHiddenToOutput[row][col] * weightsHiddenToOutputModification[row][col] * errorsArray[col]

        self.weightsHiddenToOutput = modifyWeightsHiddenToOutput
        print(modifyWeightsHiddenToOutput)


        # modifySigmoidSumOfWeightsInputProduct = np.copy(self.sigmoidSumOfWeightsInputProduct)
        # print(modifySigmoidSumOfWeightsInputProduct)
        # for row in range(len(modifySigmoidSumOfWeightsInputProduct)):
        #         modifySigmoidSumOfWeightsInputProduct[row] = _sigmoid(modifySigmoidSumOfWeightsInputProduct[row], True)
        #         modifySigmoidSumOfWeightsInputProduct[row] *= weightsHiddenToOutputModification[row][col]
        #
        # print(modifySigmoidSumOfWeightsInputProduct)
        #
        # modifyWeightsInputToHidden = np.copy(self.weightsInputToHidden)
        # for row in range(len(modifyWeightsInputToHidden)):
        #     for col in range(len(modifyWeightsInputToHidden[0])):
        #         modifyWeightsInputToHidden[row][col] += modifyWeightsInputToHidden[row][col] * modifySigmoidSumOfWeightsInputProduct[col] * input[row]
        #
        # print(modifyWeightsInputToHidden)


    def trainOne(self, input, expected):
        output = self.forwardPropagation(input)
        self.backwardPropagationOld(input, expected, output, 1,
                                    None)

    def trainNew(self, input, expected, epochNum):
        errorsOfEpoch = []
        for i in range(epochNum):
            for j in range(len(input)):
                output = self.forwardPropagation(input[j])
                self.backwardPropagationOld(input[j], _castClassNamesToZerosOnesArray(expected[j]), output, epochNum, errorsOfEpoch)
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
        testingStats.append(_castClassNamesToZerosOnesArray(expected) - output)  # 7. bład na poszeczgólnych wyjsciach sieci
        testingStats.append(_sum(output, _castClassNamesToZerosOnesArray(expected)) ) # 8. popełniony przez siec bład dla całego wzorca
        return testingStats
