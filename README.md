
SkyNet is a light deep learning library. 

| **`Linux/Windows`** | **`License`** |
|------------------|------------------|
|[![Build Status](https://travis-ci.com/Tyill/skynet.svg?branch=master)](https://travis-ci.com/Tyill/skynet)|[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)|

[ResNet cpp-example for Win](https://github.com/Tyill/storage/tree/master/resnetDemo/Builds) 

<img src="https://github.com/Tyill/skynet/blob/master/docs/resnetExample.gif" width="600" height="300" />

[Compare with Tensorflow](https://github.com/Tyill/skynet/blob/master/example/resnet50/compareWithTF.py), inference ResNet50. PC: i5-2400, GF1050, Win7, MSVC12.
 
|                  | **CPU: time/img, ms** | **GPU: time/img, ms** | **CPU: RAM, Mb** | **GPU: RAM, Mb** |
|------------------|-----------------------|-----------------------|------------------|------------------|
|    Skynet        |        195            |          15           |       600        |       800        |               
|    Tensorflow    |        250            |          25           |       400        |       1400       |               

## Features

* the library is written from scratch in C++ (only STL + OpenBLAS for calculation), C-interface

* win / linux;

* network structure is set in JSON;

* base layers: fully connected, convolutional, pooling. Additional: resize, crop ..;

* basic chips: batchNorm, dropout, weight optimizers - adam, adagrad ..;

* for calculation on the CPU, OpenBLAS is used, for the video card - CUDA / cuDNN;

* for each layer there is an opportunity to separately set on what to count - CPU or GPU (and which one);

* the size of the input data is not rigidly specified, may vary in the process of work / training;

* interfaces for C++, C# and Python.


## Python example

```python

# create net
net = snNet.Net()
net.addNode('In', snOperator.Input(), 'C1') \
   .addNode('C1', snOperator.Convolution(15), 'C2') \
   .addNode('C2', snOperator.Convolution(25), 'P1') \
   .addNode('P1', snOperator.Pooling(), 'F1') \
   .addNode('F1', snOperator.FullyConnected(256), 'F2') \
   .addNode('F2', snOperator.FullyConnected(10), 'LS') \
   .addNode('LS', snOperator.LossFunction(snType.lossType.softMaxToCrossEntropy), 'Output')
   
   .............

# cycle lern
for n in range(1000):
   acc = [0]  
   net.training(lr, inLayer, outLayer, targLayer, acc)

   # calc accurate
   acc[0] = 0
   for i in range(bsz):
       if (np.argmax(outLayer[i][0][0]) == np.argmax(targLayer[i][0][0])):
           acc[0] += 1

   accuratSumm += acc[0]/bsz

   print(datetime.datetime.now().strftime('%H:%M:%S'), n, "accurate", accuratSumm / (n + 1))

```

## Install in Python

* pip install libskynet     -  CPU

* pip install libskynet-cu  -  CUDA + cuDNN7.3.1

 
## [Wiki](https://github.com/Tyill/skynet/wiki) 

## [Examples](https://github.com/Tyill/skynet/tree/master/example) 
 
## License
Licensed under an [MIT-2.0]-[license](LICENSE).
