import random

import pandas as pd
import network as nt
import numpy as np

# reading data file
data = np.genfromtxt('iris.data', delimiter=",", dtype="str")
# shuffle rows in data to get random iris population
np.random.shuffle(data)

# get iris name class
dataClass = data[:, 4]
# get iris values
dataValues = data[:, :4]
# cast values from str to float
dataValues = np.asarray(dataValues, dtype=float)

trainingValues = dataValues[:75]
trainingClasses = dataClass[:75]

testValues = dataValues[125:]
testClasses = dataClass[125:]

# mlp = nt.Network(4, 2, 3, None, None, momentum=0.8)
# mlp.trainNew(trainingValues, trainingClasses, 50)


amount = 40
zbiorTren = np.zeros((amount, 4), int)
expectedTren = np.zeros((amount, 4), int)
minV = 1
maxV = 5
for i in range(amount):
    v1 = random.randint(minV, maxV)
    v2 = random.randint(minV, maxV)
    v3 = random.randint(minV, maxV)
    v4 = random.randint(minV, maxV)
    zbiorTren[i] = [v1, v2, v3, v4]
    for x in range(4):
        if zbiorTren[i][x] == 2:
            expectedTren[i][x] = 1


mlp = nt.Network(4, 3, 4, None, None, momentum=0.8)
mlp.trainTest(zbiorTren, expectedTren, 5)

zbior = np.zeros((amount, 4), int)
for i in range(10):
    v1 = random.randint(minV, maxV)
    v2 = random.randint(minV, maxV)
    v3 = random.randint(minV, maxV)
    v4 = random.randint(minV, maxV)
    zbior[i] = [v1, v2, v3, v4]
    wynik = mlp.forwardPropagation(zbior[i])
    print(zbior[i])
    print(wynik)

# for i in range(amount):
#     print(zbior[i])
#     print(expected[i])
#     print("\n")