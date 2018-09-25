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

    def getParamsJn(self):
        return '{}'

    def name(self):
        return "Input"

class FullyConnected():
    '''Fully connected layer'''

    kernel = 0
    act = active.relu
    opt = optimizer.adam
    dropOut = 0.0
    bnorm = batchNormType.none
    mode = calcMode.CPU
    gpuDeviceId = 0
    gpuClearMem = False
    freeze = False
    wini = weightInit.he
    decayMomentDW = 0.9
    decayMomentWGr = 0.99
    lmbRegular = 0.001
    batchNormLr = 0.001

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
        self.kernel = kernel
        self.act = act
        self.opt = opt
        self.dropOut = dropOut
        self.bnorm = bnorm
        self.mode = mode
        self.gpuDeviceId = gpuDeviceId
        self.gpuClearMem = gpuClearMem
        self.freeze = freeze

    def __init__(self,
                 kernel,
                 mode=calcMode.CPU):
        self.kernel = kernel
        self.mode = mode

    def getParamsJn(self):
        ret = '{'
        ret += str(self.kernel) + ','
        ret += self.act + ','
        ret += self.opt + ','
        ret += str(self.dropOut) + ','
        ret += self.bnorm + ','
        ret += self.mode + ','
        ret += str(self.gpuDeviceId) + ','
        ret += ('1' if self.gpuClearMem else '0') + ','
        ret += ('1' if self.freeze else '0') + ','
        ret += self.wini + ','
        ret += str(self.decayMomentDW) + ','
        ret += str(self.decayMomentWGr) + ','
        ret += str(self.lmbRegular) + ','
        ret += str(self.batchNormLr) + '}'
        return ret

    def name(self):
        return "FullyConnected"
