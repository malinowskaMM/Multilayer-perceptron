import random

import network as nt
import numpy as np


def train(net, input, expected, epochNum):
    order = np.asarray([0, 1, 2, 3])
    for i in range(epochNum):
        np.random.shuffle(order)
        for j in range(4):
            output = net.forwardPropagation(input[order[j]])
            net.backwardPropagation(input[order[j]], expected[order[j]], output)


data = np.asarray(
         [[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]])

mlp = nt.Network(4, 2, 4, None, None)
train(mlp, data, data, 1000)
print("hidden neurons output:", mlp.sigmoidSumOfWeightsInputProduct)
result = mlp.forwardPropagation(data[0])
print("input:", data[0])
print("result:", result)