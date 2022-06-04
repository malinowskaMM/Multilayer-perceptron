import Layer
import numpy as np
import Sigma


def error(outputForwads, expected):
    result = 0
    length = np.size(outputForwads)
    for i in length:
        result += np.power(outputForwads[i] - expected[i], 2)
    return result / length


def errorDeriv(outputForwads, expected):
    return 2 * (outputForwads - expected) / np.size(outputForwads)


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, inputs, expected, epochs=1000, alpha=0.6):
    for i in range(epochs):
        calculatedError = 0
        for x, y in zip(inputs, expected):
            output = predict(network, x)  # go forward
            calculatedError += error(y, output)  # calculate error
            back = errorDeriv(y, output)

            for layer in reversed(network):
                back = layer.backward(back, alpha)

        calculatedError /= len(inputs)
        print("calculatedError:", calculatedError)


data = np.asarray(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])

network = [Layer(4, 2), Sigma(), Layer(2, 4), Sigma()]

train(network, data, data)

