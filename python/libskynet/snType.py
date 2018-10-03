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

from enum import Enum

class active(Enum):
    """Activation function type."""
    none = 'none'
    sigmoid = 'sigmoid'
    relu = 'relu'
    leakyRelu = 'leakyRelu'
    elu = 'elu'

class weightInit(Enum):
    """Type of initialization of weights."""
    uniform = 'uniform'
    he = 'he'
    lecun = 'lecun'
    xavier = 'xavier'

class batchNormType(Enum):
    """Type of batch norm."""
    none = 'none'
    beforeActive = 'beforeActive'
    postActive = 'postActive'

class optimizer(Enum):
    """Optimizer of weights."""
    sgd = 'sgd'
    sgdMoment = 'sgdMoment'
    adagrad = 'adagrad'
    RMSprop = 'RMSprop'
    adam = 'adam'

class optimizer(Enum):
    """Optimizer of weights."""
    sgd = 'sgd'
    sgdMoment = 'sgdMoment'
    adagrad = 'adagrad'
    RMSprop = 'RMSprop'
    adam = 'adam'

class poolType(Enum):
    """Pooling type."""
    max = 'max'
    avg = 'avg'

class calcMode(Enum):
    """Calc mode."""
    CPU = 'CPU'
    CUDA = 'CUDA'
    #OpenCL = 'OpenCL'

class lockType(Enum):
    """Lock type."""
    lock = 'lock'
    unlock = 'unlock'

class summatorType(Enum):
    """Summator type."""
    summ = 'summ'
    diff = 'diff'
    mean = 'mean'

class lossType(Enum):
    """Loss type."""
    softMaxToCrossEntropy = 'softMaxToCrossEntropy'
    binaryCrossEntropy = 'binaryCrossEntropy'
    regressionOLS = 'regressionOLS'
    userLoss = 'userLoss'

class diap():
    """Diapason"""
    def __init__(self, begin: int = 0, end: int = 0):
        self.begin = begin
        self.end = end

    def value(self):
        return self.begin + ' ' + self.end

class rect():
    """Rectangle"""
    def __init__(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def value(self):
        return self.x + ' ' + self.y + ' ' + self.w + ' ' + self.h