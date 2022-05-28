
import pandas as pd
import network as nt
import numpy as np

# reading data file
data = np.genfromtxt('iris.data', delimiter=",", dtype="str")
# shuffle rows in data to get random iris population
np.random.shuffle(data)

#get iris name class
dataClass = data[:, 4]
#get iris values
dataValues = data[:, :4]
#cast values from str to float
dataValues = np.asarray(dataValues, dtype=float)


trainingValues = dataValues[:75]
trainingClasses = dataClass[:75]

testValues = dataValues[125:]
testClasses = dataClass[125:]

def castClassNamesToZerosOnesArray(className):
    if className == 'Iris-versicolor':
        return [1, 0, 0]
    elif className == 'Iris-setosa':
        return [0, 1, 0]
    elif className == 'Iris-virginica':
        return [0, 0, 1]


mlp = nt.Network(4, 5, 3, None, None)
mlp.train(trainingValues[0], castClassNamesToZerosOnesArray(trainingClasses[0]))
# for i in range(len(trainingValues)):
#     mlp.train(trainingValues[i], castClassNamesToZerosOnesArray(trainingClasses[i]))

