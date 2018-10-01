
import snNet
import snOperator
import numpy as np
import imageio
import random
import os

# create net
net = snNet.Net()
net.addNode('In', snOperator.Input(), 'F1')
net.addNode('F1', snOperator.FullyConnected(125), 'Output')


# loadImg
imgList = []
pathImg = 'c:/C++/skyNet/test/mnist/'
for i in range(10):
   imgList.append(os.listdir(pathImg + str(i)))

bsz = 100
inLayer = np.arange(28*28*bsz, float).reshape(bsz, 28, 28)
outLayer = np.arange(10*bsz, float).reshape(bsz, 10)
targLayer = np.arange(10*bsz, float).reshape(bsz, 10)

# cycle
for n in range(100):
    for i in range(bsz):
        ndir = random.randint(0, 10)
        nimg = random.randint(0, len(imgList[ndir]))
        im = imageio.imread(pathImg + str(ndir) + '/' + imgList[ndir][nimg])

        inLayer

    acc = [0]
    net.training(0.001, inLayer, outLayer, targLayer, acc)

    print(acc[0])


