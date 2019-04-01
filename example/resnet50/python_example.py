
from libskynet import*
import numpy as np
from keras.preprocessing import image


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


### Create net
net = snNet.Net()

net.addNode('In', snOperator.Input(), 'conv1') \
   .addNode('conv1', snOperator.Convolution(64, (7, 7), 3, 2, snType.batchNormType.none), 'pool1_pad') \
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

#################################


### Set weight

weightTF = snWeight.loadHdf5Group('c:\\Users\\a.medvedev\\.keras\\models\\resnet50_weights_tf_dim_ordering_tf_kernels.h5')

def setWeightNd(net, nm: str, w: np.ndarray, b: np.ndarray):
    if (w.ndim == 4):
        w.resize((w.shape[0] * w.shape[1] * w.shape[2], w.shape[3]))
    wb = np.vstack((w, b))
    wb.resize((1, 1, wb.shape[0], wb.shape[1]))
    net.setWeightNode(nm, wb)

def setBNormNd(net, nm: str, bn: np.ndarray, outSz : ()):

    tmp = np.zeros((bn[0].shape[0], outSz[0], outSz[1]))

    newBn = np.zeros((4, bn[0].shape[0], outSz[0], outSz[1]))

    for j in range(4):
      for i in range(bn[0].shape[0]):
        tmp[i] = np.full((outSz[0], outSz[1]), bn[j][i])
      newBn[j] = tmp

    net.setBNormNode(nm, [newBn[3], newBn[0], newBn[2], newBn[1]])

    return

def setConvWeight(net, wName, bnName, weight, outSz : (), isIdnt = True):
    wnm = wName + '2a'
    bnm = bnName + '2a'
    setWeightNd(net, wnm, weight[wnm][0], weight[wnm][1])
    setBNormNd(net, wnm, weight[bnm], outSz)

    wnm = wName + '2b'
    bnm = bnName + '2b'
    setWeightNd(net, wnm, weight[wnm][0], weight[wnm][1])
    setBNormNd(net, wnm, weight[bnm], outSz)

    wnm = wName + '2c'
    bnm = bnName + '2c'
    setWeightNd(net, wnm, weight[wnm][0], weight[wnm][1])
    setBNormNd(net, wnm, weight[bnm], outSz)

    if (not isIdnt):
       wnm = wName + '1'
       bnm = bnName + '1'
       setWeightNd(net, wnm, weight[wnm][0], weight[wnm][1])
       setBNormNd(net, wnm, weight[bnm], outSz)

    return


setWeightNd(net, 'conv1', weightTF['conv1'][0], weightTF['conv1'][1])
setBNormNd(net, 'conv1', weightTF['bn_conv1'], (112, 112))

setConvWeight(net, 'res2a_branch', 'bn2a_branch', weightTF, (56, 56), False)
setConvWeight(net, 'res2b_branch', 'bn2b_branch', weightTF, (56, 56), True)
setConvWeight(net, 'res2c_branch', 'bn2c_branch', weightTF, (56, 56), True)

setConvWeight(net, 'res3a_branch', 'bn3a_branch', weightTF, (28, 28), False)
setConvWeight(net, 'res3b_branch', 'bn3b_branch', weightTF, (28, 28), True)
setConvWeight(net, 'res3c_branch', 'bn3c_branch', weightTF, (28, 28), True)
setConvWeight(net, 'res3d_branch', 'bn3d_branch', weightTF, (28, 28), True)

setConvWeight(net, 'res4a_branch', 'bn4a_branch', weightTF, (14, 14), False)
setConvWeight(net, 'res4b_branch', 'bn4b_branch', weightTF, (14, 14), True)
setConvWeight(net, 'res4c_branch', 'bn4c_branch', weightTF, (14, 14), True)
setConvWeight(net, 'res4d_branch', 'bn4d_branch', weightTF, (14, 14), True)
setConvWeight(net, 'res4e_branch', 'bn4e_branch', weightTF, (14, 14), True)
setConvWeight(net, 'res4f_branch', 'bn4f_branch', weightTF, (14, 14), True)

setConvWeight(net, 'res5a_branch', 'bn5a_branch', weightTF, (7, 7), False)
setConvWeight(net, 'res5b_branch', 'bn5b_branch', weightTF, (7, 7), True)
setConvWeight(net, 'res5c_branch', 'bn5c_branch', weightTF, (7, 7), True)

setWeightNd(net, 'fc1000', weightTF['fc1000'][0], weightTF['fc1000'][1])

#################################


img_path = 'c:\\cpp\\101_ObjectCategories\\panda\\image_0001.jpg'

img = image.load_img(img_path, target_size=(224, 224))
inAr = image.img_to_array(img)
#inAr = np.transpose(inAr, (2, 1, 0))
inAr = inAr.reshape(1, 3, 224, 224)

outAr = np.zeros((1, 1, 1, 1000), ctypes.c_float)

net.forward(False, inAr, outAr)

mx = np.argmax(outAr[0])

print('Predicted:', mx, 'val', outAr[0][0][0][mx])

## Compare with TF
#from keras.applications.resnet50 import ResNet50
#from keras.applications.resnet50 import preprocess_input, decode_predictions
#
#model = ResNet50(weights='imagenet')
#
#
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
#
# mx = np.argmax(preds[0])
#
# print('Predicted:', mx, 'val', preds[0][mx])

#print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]




