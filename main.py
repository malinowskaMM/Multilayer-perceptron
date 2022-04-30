import numpy as np


class Network:
    def __init__(self, inputNumber, hiddenNumber, outputNumber):
        self.inputNumber = inputNumber
        self.hiddenNumber = hiddenNumber
        self.outputNumber = outputNumber

        layers = [self.inputNumber] + self.hiddenNumber + [self.outputNumber]

        # init random weights
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

    def forwardPropagation(self, input):
        activation = input

        for w in self.weights:
            networkInputs = np.dot(activation, w)

            activation = self._sigmoid(networkInputs)

        return activation

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
