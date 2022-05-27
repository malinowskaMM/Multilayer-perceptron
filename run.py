'''
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


trainingValues = dataValues[:25]
trainingClasses = dataClass[:25]
'''


'''input = np.array([1, 2])
net = nt.Network(2, 2, 2, None, None)
for i in range(3):
    net.train(input, np.array([2, 6]))'''

input = np.array([1, 2, 3, 4])
net = nt.Network(4, 2, 1, None, None)
for i in range(6):
    net.train(input, 1)

print("----------------------")
#res = net.forwardPropagation(input)
#print(res)