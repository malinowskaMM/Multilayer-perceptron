import pandas as pd
import network as nt
import numpy as np

# reading csv files
data = pd.read_csv('iris.data', sep=",")
# print(data)

input = np.array([1, 2])
net = nt.Network(2, 2, 1, None, None)
for i in range(3):
    net.train(input, 3)

res = net.forwardPropagation(input)
print(res)