
from libskynet import snOperator, snType, snNet
import numpy as np

######## Create net

def _idntBlock(net,
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
                                                       snType.batchNormType.beforeActive,
                                                       snType.active.relu),
                                                       oprName + '2b') \
       .addNode(oprName + '2b', snOperator.Convolution(filters[1],
                                                       kernelSize,
                                                       -1,  # same padding
                                                       1,
                                                       snType.batchNormType.beforeActive,
                                                       snType.active.relu),
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


def _convBlock(net,
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
                                                       snType.batchNormType.beforeActive,
                                                       snType.active.relu),
                                                       oprName + '2b') \
       .addNode(oprName + '2b', snOperator.Convolution(filters[1],
                                                       kernelSize,
                                                       -1,  # same padding
                                                       1,
                                                       snType.batchNormType.beforeActive,
                                                       snType.active.relu),
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


def createNet():
    """
    Create net
    :return: net
    """

    net = snNet.Net()

    net.addNode('In', snOperator.Input(), 'conv1') \
       .addNode('conv1',
                snOperator.Convolution(64, (7, 7), 3, 2, snType.batchNormType.beforeActive, snType.active.none),
                'pool1_pad') \
       .addNode('pool1_pad', snOperator.Pooling(3, 2, snType.poolType.max), 'res2a_branch1 res2a_branch2a')

    _convBlock(net, (3, 3), filters=[64, 64, 256], stride=1, oprName='res2a_branch',
              nextOprName='res2b_branch2a res2b_branchSum')
    _idntBlock(net, (3, 3), filters=[64, 64, 256], oprName='res2b_branch',
               nextOprName='res2c_branch2a res2c_branchSum')
    _idntBlock(net, (3, 3), filters=[64, 64, 256], oprName='res2c_branch',
               nextOprName='res3a_branch1 res3a_branch2a')

    _convBlock(net, (3, 3), filters=[128, 128, 512], stride=2, oprName='res3a_branch',
              nextOprName='res3b_branch2a res3b_branchSum')
    _idntBlock(net, (3, 3), filters=[128, 128, 512], oprName='res3b_branch',
              nextOprName='res3c_branch2a res3c_branchSum')
    _idntBlock(net, (3, 3), filters=[128, 128, 512], oprName='res3c_branch',
              nextOprName='res3d_branch2a res3d_branchSum')
    _idntBlock(net, (3, 3), filters=[128, 128, 512], oprName='res3d_branch',
              nextOprName='res4a_branch1 res4a_branch2a')

    _convBlock(net, (3, 3), filters=[256, 256, 1024], stride=2, oprName='res4a_branch',
              nextOprName='res4b_branch2a res4b_branchSum')
    _idntBlock(net, (3, 3), filters=[256, 256, 1024], oprName='res4b_branch',
              nextOprName='res4c_branch2a res4c_branchSum')
    _idntBlock(net, (3, 3), filters=[256, 256, 1024], oprName='res4c_branch',
              nextOprName='res4d_branch2a res4d_branchSum')
    _idntBlock(net, (3, 3), filters=[256, 256, 1024], oprName='res4d_branch',
              nextOprName='res4e_branch2a res4e_branchSum')
    _idntBlock(net, (3, 3), filters=[256, 256, 1024], oprName='res4e_branch',
              nextOprName='res4f_branch2a res4f_branchSum')
    _idntBlock(net, (3, 3), filters=[256, 256, 1024], oprName='res4f_branch',
              nextOprName='res5a_branch1 res5a_branch2a')

    _convBlock(net, (3, 3), filters=[512, 512, 2048], stride=2, oprName='res5a_branch',
              nextOprName='res5b_branch2a res5b_branchSum')
    _idntBlock(net, (3, 3), filters=[512, 512, 2048], oprName='res5b_branch',
              nextOprName='res5c_branch2a res5c_branchSum')
    _idntBlock(net, (3, 3), filters=[512, 512, 2048], oprName='res5c_branch',
              nextOprName='avg_pool')

    net.addNode('avg_pool', snOperator.Pooling(7, 7, snType.poolType.avg), 'fc1000') \
        .addNode('fc1000', snOperator.FullyConnected(1000, snType.active.none), 'LS') \
        .addNode('LS', snOperator.LossFunction(snType.lossType.softMaxToCrossEntropy), 'Output')

    return net



####### Set weights

def _setWeightNd(net, nm: str, w: np.ndarray, b: np.ndarray) -> bool:
    """
    setWeightNd
    :param net:
    :param nm:
    :param w:
    :param b:
    :return: true - ok
    """
    if (w.ndim == 4):
        w = np.moveaxis(w, -1, 0)
        w = np.moveaxis(w, -1, 1).copy()
        w = w.reshape((w.shape[1] * w.shape[2] * w.shape[3], w.shape[0]))

        wb = np.vstack((w, b))
        wb.resize((1, 1, wb.shape[0], wb.shape[1]))

        ok = net.setWeightNode(nm, wb)

    else:  # ndim == 2
        wb = np.vstack((w, b)).copy()
        wb.resize((1, 1, wb.shape[0], wb.shape[1]))
        ok = net.setWeightNode(nm, wb)

    return ok

def _setBNormNd(net, nm: str, bn: [], outSz: ()) -> bool:
    """
    setBNormNd
    :param net:
    :param nm:
    :param bn:  gamma, betta, mean, varce
    :param outSz:
    :return: true - ok
    """

    tmp = np.zeros((bn[0].shape[0], outSz[0], outSz[1]), np.float32)

    newBn = np.zeros((4, bn[0].shape[0], outSz[0], outSz[1]), np.float32)

    bn[3] = (bn[3] + 0.001) ** 0.5

    for j in range(4):
        for i in range(bn[0].shape[0]):
            tmp[i] = np.full((outSz[0], outSz[1]), bn[j][i])
        newBn[j] = tmp

    ok = net.setBNornNode(nm, (newBn[0], newBn[1], newBn[2], newBn[3]))

    return ok

def _setConvWeight(net, wName, bnName, weight, outSz: (), isIdnt=True) -> bool:
    """
    _setConvWeight
    :param net:
    :param wName:
    :param bnName:
    :param weight:
    :param outSz:
    :param isIdnt:
    :return:
    """

    ok = 1

    wnm = wName + '2a'
    bnm = bnName + '2a'
    ok &= _setWeightNd(net, wnm, weight[wnm][0], weight[wnm][1])
    ok &= _setBNormNd(net, wnm, weight[bnm], outSz)

    wnm = wName + '2b'
    bnm = bnName + '2b'
    ok &= _setWeightNd(net, wnm, weight[wnm][0], weight[wnm][1])
    ok &= _setBNormNd(net, wnm, weight[bnm], outSz)

    wnm = wName + '2c'
    bnm = bnName + '2c'
    ok &= _setWeightNd(net, wnm, weight[wnm][0], weight[wnm][1])
    ok &= _setBNormNd(net, wnm, weight[bnm], outSz)

    if (not isIdnt):
        wnm = wName + '1'
        bnm = bnName + '1'
        ok &= _setWeightNd(net, wnm, weight[wnm][0], weight[wnm][1])
        ok &= _setBNormNd(net, wnm, weight[bnm], outSz)

    return ok

def setWeights(net, weightTF) -> bool:
    """
    setWeights
    :param net: skynet
    :param weightTF: weight TF
    :return: true - ok
    """

    ok = 1

    ok &= _setWeightNd(net, 'conv1', weightTF['conv1'][0], weightTF['conv1'][1])
    ok &= _setBNormNd(net, 'conv1', weightTF['bn_conv1'], (112, 112))

    ok &= _setConvWeight(net, 'res2a_branch', 'bn2a_branch', weightTF, (56, 56), False)
    ok &= _setConvWeight(net, 'res2b_branch', 'bn2b_branch', weightTF, (56, 56), True)
    ok &= _setConvWeight(net, 'res2c_branch', 'bn2c_branch', weightTF, (56, 56), True)

    ok &= _setConvWeight(net, 'res3a_branch', 'bn3a_branch', weightTF, (28, 28), False)
    ok &= _setConvWeight(net, 'res3b_branch', 'bn3b_branch', weightTF, (28, 28), True)
    ok &= _setConvWeight(net, 'res3c_branch', 'bn3c_branch', weightTF, (28, 28), True)
    ok &= _setConvWeight(net, 'res3d_branch', 'bn3d_branch', weightTF, (28, 28), True)

    ok &= _setConvWeight(net, 'res4a_branch', 'bn4a_branch', weightTF, (14, 14), False)
    ok &= _setConvWeight(net, 'res4b_branch', 'bn4b_branch', weightTF, (14, 14), True)
    ok &= _setConvWeight(net, 'res4c_branch', 'bn4c_branch', weightTF, (14, 14), True)
    ok &= _setConvWeight(net, 'res4d_branch', 'bn4d_branch', weightTF, (14, 14), True)
    ok &= _setConvWeight(net, 'res4e_branch', 'bn4e_branch', weightTF, (14, 14), True)
    ok &= _setConvWeight(net, 'res4f_branch', 'bn4f_branch', weightTF, (14, 14), True)

    ok &= _setConvWeight(net, 'res5a_branch', 'bn5a_branch', weightTF, (7, 7), False)
    ok &= _setConvWeight(net, 'res5b_branch', 'bn5b_branch', weightTF, (7, 7), True)
    ok &= _setConvWeight(net, 'res5c_branch', 'bn5c_branch', weightTF, (7, 7), True)

    ok &= _setWeightNd(net, 'fc1000', weightTF['fc1000'][0], weightTF['fc1000'][1])

    return ok
