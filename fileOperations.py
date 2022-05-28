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
    file.writelines(line)

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


def loadNetworkFromFile(name):
    file = open(name, 'r')
    fileContent = file.readlines()
    numbers = fileContent[0].split(",")
    numbers = numbers[:3]
    numbers = [int(x) for x in numbers]

    weightsInputToHiddenInput = fileContent[1].split(",")
    weightsInputToHiddenInput = weightsInputToHiddenInput[:len(weightsInputToHiddenInput)-1]
    weightsInputToHiddenInput = [float(x) for x in weightsInputToHiddenInput]

    # weightsInputToHidden (inputNumber, hiddenNumber)
    weightsInputToHidden = np.zeros((numbers[0], numbers[1]))
    for row in range(len(weightsInputToHidden)):
        for col in range(len(weightsInputToHidden[0])):
            weightsInputToHidden[row][col] = weightsInputToHiddenInput[col + row * col]

    weightsHiddenToOutputInput = fileContent[2].split(",")
    weightsHiddenToOutputInput = weightsHiddenToOutputInput[:len(weightsHiddenToOutputInput)-1]
    weightsHiddenToOutputInput = [float(x) for x in weightsHiddenToOutputInput]

    # weightsHiddenToOutput (hiddenNumber, outputNumber)
    weightsHiddenToOutput = np.zeros((numbers[1], numbers[2]))
    for row in range(len(weightsHiddenToOutput)):
        for col in range(len(weightsHiddenToOutput[0])):
            weightsHiddenToOutput[row][col] = weightsHiddenToOutputInput[col + row * col]

    mlp = network.Network(numbers[0], numbers[1], numbers[2], weightsInputToHidden, weightsHiddenToOutput)
    return mlp


def writeErrorsOfEpochToFile(valuesList):
    time = datetime.datetime.now()
    file = open(
        'mlp_learn_stats_' + "%s_%s_%s_%s_%s_%s.txt" % (time.day, time.month, time.year, time.hour, time.minute, time.second), "w")
    valuesStr=""
    for value in valuesList:
        valuesStr += str(value) + ","
    file.write(valuesStr)


def loadData(name, inputsNumber):
    # reading data file
    data = np.genfromtxt(name, delimiter=",", dtype="str")
    # shuffle rows in data to get random iris population
    np.random.shuffle(data)

    # get iris name class
    dataClass = data[:, inputsNumber]
    # get iris values
    dataValues = data[:, :inputsNumber]
    # cast values from str to float
    dataValues = np.asarray(dataValues, dtype=float)

    return [dataValues, dataClass]