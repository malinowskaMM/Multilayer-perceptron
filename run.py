import numpy as np

import network as nt
import fileOperations

data = fileOperations.loadData('iris.data', 4)

trainingValues = data[0][:75]
trainingClasses = data[1][:75]


mlp = nt.Network(4, 2, 3)
fileOperations.writeErrorsOfEpochToFile(mlp.trainNew(trainingValues, trainingClasses, 50))
test = np.array([1, 2, 3, 4])
exp = np.array([0.3, 0.5, 0.7])
result = mlp.testingOne(test, exp)
fileOperations.writeTestingStats(result)

# mlp2 = fileOperations.loadNetworkFromFile('mlp_28_5_2022_19_59_59.txt')
# mlp2.trainNew(trainingValues, trainingClasses, 50)

