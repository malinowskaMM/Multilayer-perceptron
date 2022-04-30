import numpy as np


class Network:
    # initialize network with number of inputs, number of hidden layer (table [3,2] - 3 neurons in first hidden layer,
    # 2 neurons in second hidden layer), number of outputs
    def __init__(self, inputNumber, hiddenNumber, outputNumber):
        self.inputNumber = inputNumber
        self.hiddenNumber = hiddenNumber
        self.outputNumber = outputNumber

        # (4-elements)list which content is number of neurons in each layer
        layers = [self.inputNumber] + self.hiddenNumber + [self.outputNumber]

        # init random weights in range (-1, 1)
        # list of lists where every list in the big list contains weights for its neurons
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.uniform(-1, 1, (layers[i], layers[i+1]))
            self.weights.append(w)


    def forwardPropagation(self, input):
        activation = input
        #w is a list of weights from one layer
        for w in self.weights:
            # matrix multiplication [inputs] * [matrix of weights]
            networkInputs = np.dot(activation, w)
            activation = self._sigmoid(networkInputs)
        return activation


    # from wikipedia
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))


if __name__ == '__main__':
    # create network
    network = Network(3, [3, 5], 2)
    # create inputs
    inputs = np.random.rand(network.inputNumber)
    #forward propagataion
    outputs = network.forwardPropagation(inputs)
    print(inputs)
    print(outputs)