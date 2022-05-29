import numpy as np

def _castClassNamesToZerosOnesArray(className):
    if className == 'Iris-versicolor':
        return [1, 0, 0]
    elif className == 'Iris-setosa':
        return [0, 1, 0]
    elif className == 'Iris-virginica':
        return [0, 0, 1]

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

    def forwardPropagation(self, input, biasH=0, biasO=0):
        #single row contains weights for one hidden neuron
        productOfWeightsAndInput = np.copy(self.weightsInputToHidden)
        for row in range(len(productOfWeightsAndInput)):
            productOfWeightsAndInput[row] *= input[row]
            productOfWeightsAndInput[row] += biasH

        #sum of hidden neuron is a sum of values in single column
        self.sumOfWeightsInputProduct = np.sum(productOfWeightsAndInput, axis=0)  # vector of inputs into hidden neurons

        #for each sum we use sigmoid function
        self.sigmoidSumOfWeightsInputProduct = _sigmoid(self.sumOfWeightsInputProduct)  # vector of outputs from hidden neurons

        #single row contains weights for one output neuron
        productOfWeightsAndHidden = np.copy(self.weightsHiddenToOutput)
        for elementIter in range(len(productOfWeightsAndHidden)):
            productOfWeightsAndHidden[elementIter] *= self.sigmoidSumOfWeightsInputProduct[elementIter]
            productOfWeightsAndHidden[elementIter] += biasO
        # sum of output neuron is a sum of values in single column
        self.sumOfWeightsHiddenProduct = np.sum(productOfWeightsAndHidden, axis=0)  # vector of inputs into output neurons

        # for each sum we use sigmoid function
        sigmoidSumOfWeightsHiddenProduct = _sigmoid(self.sumOfWeightsHiddenProduct) # vector of outputs from output neurons

        output = sigmoidSumOfWeightsHiddenProduct
        return output


    #alpha - step length coefficient
    def backwardPropagation(self, input, expected, outputForward, alpha = 0.6):
        #calculate sum of errors (global error) between expected and output
        errorSum = _sum(outputForward, expected)  #Cost function
        print("errorSum:", errorSum)

        #weights hidden to output modifications
        weightsHtOBeforeChange = np.copy(self.weightsHiddenToOutput)  # needed for next layer calculations
        modifyHiddenToOutputWeights = np.zeros((self.hiddenNumber, self.outputNumber))
        # modifyHiddenToOutputWeights :  rows = hidden, col = output
        for row in range(self.hiddenNumber):
            for col in range(self.outputNumber):
                change = -1 * (expected[col] - outputForward[col]) * outputForward[col] * (1 - outputForward[col]) * self.weightsHiddenToOutput[row][col]
                self.weightsHiddenToOutput[row][col] -= alpha * change  # weight modification (hidden to output)


        #weights input to hidden modifications
        modifyInputToHiddenWeights = np.copy(self.weightsInputToHidden)
        # modifyInputToHiddenWeights :  rows = input, col = hidden

        # weights used here are weightsHtOBeforeChange, NOT self.weightsHiddenToOutput (which is modified at this point)
        for row in range(self.inputNumber):
            for col in range(self.hiddenNumber):
                change = 0
                for outputIter in range(self.outputNumber):
                    x = -1 * (expected[outputIter] - outputForward[outputIter]) * outputForward[outputIter] * (1 - outputForward[outputIter]) * self.weightsInputToHidden[row][col]
                    y = self.sumOfWeightsInputProduct[col] * (1 - self.sumOfWeightsInputProduct[col])
                    z = input[row]
                    self.weightsInputToHidden[row][col] -= alpha * x * y * z


    def trainNew(self, input, expected, epochNum):
        errorsOfEpoch = []
        for i in range(epochNum):
            for j in range(len(input)):
                output = self.forwardPropagation(input[j], -0.6, -0.6)
                self.backwardPropagation(input[j], _castClassNamesToZerosOnesArray(expected[j]), output)
                # print("hw:",self.weightsInputToHidden)
        return errorsOfEpoch

