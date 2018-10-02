#
# SkyNet Project
# Copyright (C) 2018 by Contributors <https:#github.com/Tyill/skynet>
#
# This code is licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from snType import *

class Input():
    """Input layer."""

    def __init__(self):
        pass

    def getParams(self):
        return {}

    def name(self):
        return 'Input'

class FullyConnected():
    '''Fully connected layer'''

    _params = {
    'kernel' : '0',
    'act' : active.relu.value,
    'opt' : optimizer.adam.value,
    'dropOut' : '0',
    'bnorm' : batchNormType.none.value,
    'mode' : calcMode.CPU.value,
    'gpuDeviceId' : '0',
    'gpuClearMem' : '0',
    'freeze' :'0',
    'wini' : weightInit.he.value,
    'decayMomentDW' : '0.9',
    'decayMomentWGr' : '0.99',
    'lmbRegular' : '0.001',
    'batchNormLr' : '0.001'
    }

    def __init__(self,
                 kernel,
                 act=active.relu,
                 opt=optimizer.adam,
                 dropOut=0.0,
                 bnorm=batchNormType.none,
                 mode=calcMode.CPU,
                 gpuDeviceId=0,
                 gpuClearMem=False,
                 freeze=False):
        self._params['kernel'] = str(kernel)
        self._params['act'] = act.value
        self._params['opt'] = opt.value
        self._params['dropOut'] = str(dropOut)
        self._params['bnorm'] = bnorm.value
        self._params['mode'] = mode.value
        self._params['gpuDeviceId'] = str(gpuDeviceId)
        self._params['gpuClearMem'] = '1' if gpuClearMem else '0'
        self._params['freeze'] = '1' if freeze else '0'

    def __init__(self,
                 kernel,
                 mode=calcMode.CPU):
        self._params['kernel'] = str(kernel)
        self._params['mode'] = mode.value

    def getParams(self):
        return self._params

    def name(self):
        return "FullyConnected"

class Convolution():
    '''Convolution layer'''

    _params = {
    'kernel' : '0',
    'fWidth': '3',
    'fHeight': '3',
    'padding': '0',
    'stride':'1',
    'dilate': '1',
    'act' : active.relu.value,
    'opt' : optimizer.adam.value,
    'dropOut' : '0',
    'bnorm' : batchNormType.none.value,
    'mode' : calcMode.CPU.value,
    'gpuDeviceId' : '0',
    'gpuClearMem' : '0',
    'freeze' :'0',
    'wini' : weightInit.he.value,
    'decayMomentDW' : '0.9',
    'decayMomentWGr' : '0.99',
    'lmbRegular' : '0.001',
    'batchNormLr' : '0.001'
    }

    def __init__(self,
                 kernel,
                 act=active.relu,
                 opt=optimizer.adam,
                 dropOut=0.0,
                 bnorm=batchNormType.none,
                 mode=calcMode.CPU,
                 gpuDeviceId=0,
                 gpuClearMem=False,
                 freeze=False):
        self._params['kernel'] = str(kernel)
        self._params['act'] = act.value
        self._params['opt'] = opt.value
        self._params['dropOut'] = str(dropOut)
        self._params['bnorm'] = bnorm.value
        self._params['mode'] = mode.value
        self._params['gpuDeviceId'] = str(gpuDeviceId)
        self._params['gpuClearMem'] = '1' if gpuClearMem else '0'
        self._params['freeze'] = '1' if freeze else '0'

    def __init__(self,
                 kernel,
                 mode=calcMode.CPU):
        self._params['kernel'] = str(kernel)
        self._params['mode'] = mode.value

    def getParams(self):
        return self._params

    def name(self):
        return "Convolution"

class Deconvolution():
    '''Deconvolution layer'''

    _params = {
        'kernel': '0',
        'fWidth': '3',
        'fHeight': '3',
        'stride': '2',
        'act': active.relu.value,
        'opt': optimizer.adam.value,
        'dropOut': '0',
        'bnorm': batchNormType.none.value,
        'mode': calcMode.CPU.value,
        'gpuDeviceId': '0',
        'gpuClearMem': '0',
        'freeze': '0',
        'wini': weightInit.he.value,
        'decayMomentDW': '0.9',
        'decayMomentWGr': '0.99',
        'lmbRegular': '0.001',
        'batchNormLr': '0.001'
    }

    def __init__(self,
                 kernel,
                 act=active.relu,
                 opt=optimizer.adam,
                 dropOut=0.0,
                 bnorm=batchNormType.none,
                 mode=calcMode.CPU,
                 gpuDeviceId=0,
                 gpuClearMem=False,
                 freeze=False):
        self._params['kernel'] = str(kernel)
        self._params['act'] = act.value
        self._params['opt'] = opt.value
        self._params['dropOut'] = str(dropOut)
        self._params['bnorm'] = bnorm.value
        self._params['mode'] = mode.value
        self._params['gpuDeviceId'] = str(gpuDeviceId)
        self._params['gpuClearMem'] = '1' if gpuClearMem else '0'
        self._params['freeze'] = '1' if freeze else '0'

    def __init__(self,
                 kernel,
                 mode=calcMode.CPU):
        self._params['kernel'] = str(kernel)
        self._params['mode'] = mode.value

    def getParams(self):
        return self._params

    def name(self):
        return "Deconvolution"

class Pooling():
    '''Pooling layer'''

    _params = {
        'kernel': '0',
        'pool': poolType.max.value,
        'mode': calcMode.CPU.value,
        'gpuDeviceId': '0',
        'gpuClearMem': '0',
    }

    def __init__(self,
                 kernel,
                 mode=calcMode.CPU):
        self._params['kernel'] = str(kernel)
        self._params['mode'] = mode.value

    def getParams(self):
        return self._params

    def name(self):
        return "Pooling"

class LossFunction():
    '''Loss Function layer'''

    _params = {
        'loss': lossType.softMaxToCrossEntropy.value
    }

    def __init__(self, loss):
        self._params['loss'] = loss.value

    def getParams(self):
        return self._params

    def name(self):
        return "LossFunction"
