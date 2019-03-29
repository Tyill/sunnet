
from libskynet import*
import numpy as np
import ctypes

def idntBlock(net,
              kernelSize,
              filters,
              oprName,
              nextOprName):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        kernelSize: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path

    # Returns
        Output tensor for the block.
    """

    net.addNode(oprName + '2a', snOperator.Convolution(filters[0],
                                                       (1, 1),
                                                       0,  # no padding
                                                       1,
                                                       snType.batchNormType.beforeActive),
                                                       oprName + '2b') \
       .addNode(oprName + '2b', snOperator.Convolution(filters[1],
                                                       kernelSize,
                                                       -1,  # same padding
                                                       1,
                                                       snType.batchNormType.beforeActive),
                                                       oprName + '2c') \
       .addNode(oprName + '2c', snOperator.Convolution(filters[2],
                                                       (1, 1),
                                                       0,  # no padding
                                                       1,
                                                       snType.batchNormType.beforeActive,
                                                       snType.active.none),
                                                       oprName + 'Sum') \
    # summator
    net.addNode(oprName + 'Sum', snOperator.Summator(snType.summatorType.summ), oprName + 'Act') \
       .addNode(oprName + 'Act', snOperator.Activation(snType.active.relu), nextOprName)


def convBlock(net,
               kernelSize,
               filters,
               stride,
               oprName,
               nextOprName):
    """A block that has a conv layer at shortcut.

    # Arguments
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.
    """

    net.addNode(oprName + '2a', snOperator.Convolution(filters[0],
                                                       (1, 1),
                                                       0,  # no padding
                                                       stride,
                                                       snType.batchNormType.beforeActive),
                                                       oprName + '2b') \
       .addNode(oprName + '2b', snOperator.Convolution(filters[1],
                                                       kernelSize,
                                                       -1,  # same padding
                                                       1,
                                                       snType.batchNormType.beforeActive),
                                                       oprName + '2c') \
       .addNode(oprName + '2c', snOperator.Convolution(filters[2],
                                                       (1, 1),
                                                       0,  # no padding
                                                       1,
                                                       snType.batchNormType.beforeActive,
                                                       snType.active.none),
                                                       oprName + 'Sum') \
    # shortcut
    net.addNode(oprName + '1', snOperator.Convolution(filters[2],
                                                      (1, 1),
                                                      0,      # no padding
                                                      stride,
                                                      snType.batchNormType.beforeActive,
                                                      snType.active.none),
                                                      oprName + 'Sum') \
    # summator
    net.addNode(oprName + 'Sum', snOperator.Summator(snType.summatorType.summ), oprName + 'Act') \
       .addNode(oprName + 'Act', snOperator.Activation(snType.active.relu), nextOprName)

# create net
net = snNet.Net()

net.addNode('In', snOperator.Input(), 'conv1') \
   .addNode('conv1', snOperator.Convolution(64, (7, 7), 3, 2, snType.batchNormType.beforeActive), 'pool1_pad') \
   .addNode('pool1_pad', snOperator.Pooling(3, 2), 'res2a_branch1 res2a_branch2a')

convBlock(net, (3, 3), filters=[64, 64, 256], stride=1, oprName='res2a_branch', nextOprName='res2b_branch2a res2b_branchSum')
idntBlock(net, (3, 3), filters=[64, 64, 256], oprName='res2b_branch', nextOprName='res2c_branch2a res2c_branchSum')
idntBlock(net, (3, 3), filters=[64, 64, 256], oprName='res2c_branch', nextOprName='res3a_branch1 res3a_branch2a')

convBlock(net, (3, 3), filters=[128, 128, 512], stride=2, oprName='res3a_branch', nextOprName='res3b_branch2a res3b_branchSum')
idntBlock(net, (3, 3), filters=[128, 128, 512], oprName='res3b_branch', nextOprName='res3c_branch2a res3c_branchSum')
idntBlock(net, (3, 3), filters=[128, 128, 512], oprName='res3c_branch', nextOprName='res3d_branch2a res3d_branchSum')
idntBlock(net, (3, 3), filters=[128, 128, 512], oprName='res3d_branch', nextOprName='res4a_branch1 res4a_branch2a')

convBlock(net, (3, 3), filters=[256, 256, 1024], stride=2, oprName='res4a_branch', nextOprName='res4b_branch2a res4b_branchSum')
idntBlock(net, (3, 3), filters=[256, 256, 1024], oprName='res4b_branch', nextOprName='res4c_branch2a res4c_branchSum')
idntBlock(net, (3, 3), filters=[256, 256, 1024], oprName='res4c_branch', nextOprName='res4d_branch2a res4d_branchSum')
idntBlock(net, (3, 3), filters=[256, 256, 1024], oprName='res4d_branch', nextOprName='res4e_branch2a res4e_branchSum')
idntBlock(net, (3, 3), filters=[256, 256, 1024], oprName='res4e_branch', nextOprName='res4f_branch2a res4f_branchSum')
idntBlock(net, (3, 3), filters=[256, 256, 1024], oprName='res4f_branch', nextOprName='res5a_branch1 res5a_branch2a')

convBlock(net, (3, 3), filters=[512, 512, 2048], stride=2, oprName='res5a_branch', nextOprName='res5b_branch2a res5b_branchSum')
idntBlock(net, (3, 3), filters=[512, 512, 2048], oprName='res5b_branch', nextOprName='res5c_branch2a res5c_branchSum')
idntBlock(net, (3, 3), filters=[512, 512, 2048], oprName='res5c_branch', nextOprName='avg_pool')

net.addNode('avg_pool', snOperator.Pooling(7, 7, snType.poolType.avg), 'fc1000') \
   .addNode('fc1000', snOperator.FullyConnected(1000, snType.active.none), 'LS') \
   .addNode('LS', snOperator.LossFunction(snType.lossType.softMaxToCrossEntropy), 'Output')

inLayer = np.zeros((1, 3, 224, 224), ctypes.c_float)
outLayer = np.zeros((1, 1, 1, 1000), ctypes.c_float)

weightTF = snTF.loadHdf5Group('C:\\Users\\a.medvedev\\.keras\\models\\resnet50_weights_tf_dim_ordering_tf_kernels.h5')

net.forward(False, inLayer, outLayer)

layer = [np.ndarray]
net.getOutputNode('avg_pool', layer)

dd = True