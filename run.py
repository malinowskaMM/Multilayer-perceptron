import numpy as np

import network as nt
import fileOperations

data = fileOperations.loadData('iris.data', 4)

trainingValues = data[0][:75]
trainingClasses = data[1][:75]

weightsIToH=fileOperations.loadWeightsFromFile("weightsInputToHidden.txt")
weightsHToO=fileOperations.loadWeightsFromFile("weightsHiddenToOutput.txt")
test = np.array([1, 1, 0])
exp = np.array([1])
mlp = nt.Network(3, 2, 1,weightsIToH, weightsHToO, momentum=0.6)
mlp.trainOne(test, exp)

#print(trainingValues[0])
#print(trainingClasses[0])
#table = np.array([5.7, 2.6, 3.5, 1.0])
#fileOperations.writeErrorsOfEpochToFile(mlp.trainNew(table, "Iris-versicolor", 1))
#test = np.array([1, 2, 3, 4])
#exp = np.array([0.3, 0.5, 0.7])
#result = mlp.testingOne(data[0][-1], data[1][-1])
#fileOperations.writeTestingStats(result)

# mlp2 = fileOperations.loadNetworkFromFile('mlp_28_5_2022_19_59_59.txt')
# mlp2.trainNew(trainingValues, trainingClasses, 50)

