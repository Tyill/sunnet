
import os

from libskynet import*
import numpy as np
import imageio
import random
import ctypes
import datetime


# create net
net = snNet.Net()
net.addNode('In', snOperator.Input(), 'FC1') \
   .addNode('FC1', snOperator.FullyConnected(256), 'FC2') \
   .addNode('FC2', snOperator.FullyConnected(128), 'FC3') \
   .addNode('FC3', snOperator.FullyConnected(32), 'FC4') \
   .addNode('FC4', snOperator.FullyConnected(128), 'FC5') \
   .addNode('FC5', snOperator.FullyConnected(256), 'FC6') \
   .addNode('FC6', snOperator.FullyConnected(784), 'LS') \
   .addNode('LS', snOperator.LossFunction(snType.lossType.binaryCrossEntropy), 'Output')

# load of weight
#if (net.loadAllWeightFromFile('c:/cpp/w.dat')):
 #   print('weight is load')
#else:
#    print('error load weight')

# loadImg
imgList = []
pathImg = 'c:\\cpp\\skyNet\\example\\autoEncoder\\images\\'
for i in range(10):
   imgList.append(os.listdir(pathImg + str(i)))

bsz = 100
lr = 0.001
accuratSumm = 0.
inLayer = np.zeros((bsz, 1, 28, 28), ctypes.c_float)
outLayer = np.zeros((bsz, 1, 1, 28 * 28), ctypes.c_float)
imgMem = {}

# cycle lern
for n in range(1000):

    for i in range(bsz):
        ndir = random.randint(0, 10 - 1)
        nimg = random.randint(0, len(imgList[ndir]) - 1)

        nm = pathImg + str(ndir) + '/' + imgList[ndir][nimg]
        if (nm in imgMem):
            inLayer[i][0] = imgMem[nm]
        else:
            inLayer[i][0] = imageio.imread(nm)
            imgMem[nm] = inLayer[i][0].copy()

    acc = [0]
    net.training(lr, inLayer, outLayer, inLayer, acc)

    accuratSumm += acc[0]/bsz

    print(datetime.datetime.now().strftime('%H:%M:%S'), n, "accurate", accuratSumm / (n + 1))

# save weight
if (net.saveAllWeightToFile('c:/cpp/w.dat')):
    print('weight is save')
else:
    print('error save weight')