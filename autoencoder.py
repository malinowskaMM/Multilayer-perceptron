import random

import network as nt
import numpy as np


def trainSequentially(net, input, expected, epochNum):
    for i in range(epochNum):
        for j in range(4):
            output = net.forwardPropagation(input[j])
            # if i % 20 == 0:
            #     print(output)
            net.backwardPropagation(input[j], expected[j], output, i, None)

def train(net, input, expected, epochNum):
    order = np.asarray([0, 1, 2, 3])
    for i in range(epochNum):
        np.random.shuffle(order)
        for j in range(4):
            output = net.forwardPropagation(input[order[j]])
            if i % 20 == 0:
                print(output)
            net.backwardPropagation(input[order[j]], expected[order[j]], output, i, None)

data = np.asarray(
         [[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]])

mlp = nt.Network(4, 2, 4, momentum=0.8)
# train(mlp, data, data, 10)
trainSequentially(mlp, data, data, 1000)
print("hidden neurons output:", mlp.sigmoidSumOfWeightsInputProduct)
result = mlp.forwardPropagation(data[0])
print("input:", data[0])
print("result:", result)