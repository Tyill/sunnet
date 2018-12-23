
SkyNet is a deep learning library designed for both efficiency and flexibility. 

| **`Linux/Mac OS`** | **`Windows`** |
|------------------|-------------|
|[![Build Status](https://travis-ci.com/Tyill/skynet.svg?branch=OpenCL)](https://travis-ci.com/Tyill/skynet)|[![Build status](https://travis-ci.com/Tyill/skynet.svg?branch=OpenCL)](https://travis-ci.com/Tyill/skynet)|


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
   .addNode('C1', snOperator.Convolution(15, 0, snType.calcMode.CUDA), 'C2') \
   .addNode('C2', snOperator.Convolution(25, 0, snType.calcMode.CUDA), 'P1') \
   .addNode('P1', snOperator.Pooling(snType.calcMode.CUDA), 'F1') \
   .addNode('F1', snOperator.FullyConnected(256, snType.calcMode.CUDA), 'F2') \
   .addNode('F2', snOperator.FullyConnected(10, snType.calcMode.CUDA), 'LS') \
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

* pip install libskynet  -  CPU

* pip install libskynet-cu  -  CPU + CUDA9.2 

* pip install libskynet-cudnn   -  CPU + cuDNN7.3.1
 
 
## [Wiki](https://github.com/Tyill/skynet/wiki) 

## [Examples](https://github.com/Tyill/skynet/tree/master/example) 
 
## License
Licensed under an [MIT-2.0]-[license](LICENSE).
