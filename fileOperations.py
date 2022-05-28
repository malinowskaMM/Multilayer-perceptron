import numpy as np

import network
import datetime


# name - name of file with weights e.g. 'weightsInputToHidden.txt' or 'weightsHiddenToOutput.txt'
def loadWeightsFromFile(name):
    data = np.genfromtxt(name, delimiter=",", dtype="float")
    return data


def saveNetworkToFile(mlp: network.Network):
    time = datetime.datetime.now()
    file = open(
        'mlp_' + "%s_%s_%s_%s_%s_%s.txt" % (time.day, time.month, time.year, time.hour, time.minute, time.second), "w")
    line = f'{mlp.inputNumber},{mlp.hiddenNumber},{mlp.outputNumber},'
    file.write(line)

    lineWeightsInputToHidden = '\n'
    for row in range(len(mlp.weightsInputToHidden)):
        for col in range(len(mlp.weightsInputToHidden[0])):
            lineWeightsInputToHidden += str(mlp.weightsInputToHidden[row][col]) + ','
    file.write(lineWeightsInputToHidden)

    lineWeightsHiddenToOutput = '\n'
    for row in range(len(mlp.weightsHiddenToOutput)):
        for col in range(len(mlp.weightsHiddenToOutput[0])):
            lineWeightsHiddenToOutput += str(mlp.weightsHiddenToOutput[row][col]) + ','
    file.write(lineWeightsHiddenToOutput)
