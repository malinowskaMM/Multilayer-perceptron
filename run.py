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

mlp = nt.Network(4, 2, 3, None, None, momentum=0.8)
mlp.trainNew(trainingValues, trainingClasses, 5)
