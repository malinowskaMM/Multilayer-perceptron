import network as nt
import fileOperations

data = fileOperations.loadData('iris.data', 4)

trainingValues = data[0][:75]
trainingClasses = data[1][:75]


mlp = nt.Network(4, 2, 3, None, None)
fileOperations.writeErrorsOfEpochToFile(mlp.trainNew(trainingValues, trainingClasses, 50))


mlp2 = fileOperations.loadNetworkFromFile('mlp_28_5_2022_19_59_59.txt')
# mlp2.trainNew(trainingValues, trainingClasses, 50)

