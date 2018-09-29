
import snNet
import snOperator
from snBase import*
import numpy as np

net = snNet.Net()

net.addNode('In', snOperator.Input(), 'F1')
net.addNode('F1', snOperator.FullyConnected(125), 'Output')

a = np.arange(15).reshape(3, 5)
b = np.arange(15).reshape(3, 5)

acc = [0]
#net.forward(True,a, b)
net.training(0.001, a, b, b, acc)

a = 3


