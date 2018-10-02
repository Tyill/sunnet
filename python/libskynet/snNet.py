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
import ctypes

from __init__ import _LIB
import json
from snBase import*
import snOperator
import numpy

class Net():
    """Net object."""

    _net = 0
    _nodes = []
    _errCBack = 0

    def __init__(self, jnNet : str = '', weightPath : str = ''):

        if (len(jnNet) > 0):
            self._createNetJn(jnNet)

        if (self._net and (len(weightPath) > 0)):
            self.loadAllWeightFromFile(weightPath)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if (self._net):
            pfun = _LIB.snFreeNet
            pfun.restype = None
            pfun.argtypes = (ctypes.c_void_p)
            self._net = pfun(self._net)

    def getErrorStr(self) -> str:

        if (not self._net):
            return 'net not create'

        pfun = _LIB.snGetLastErrorStr
        pfun.restype = None
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_char_p)

        err = ctypes.create_string_buffer(256)

        self._net = pfun(self._net, err)

        return err


    def addNode(self, name : str, nd : snOperator, nextNodes : str):
        """Add node."""

        self._nodes.append({'NodeName': name, 'OperatorName': nd.name(), 'OperatorParams': nd.getParams(), 'NextNodes': nextNodes})

        return self


    def updateNode(self, name : str, nd : snOperator) -> bool:
        """Update params node."""

        ok = False
        if (self._net):
            ok = _LIB.snSetParamNode(self._net, c_str(name),  c_str(nd.getParams()))
        else:
            for n in self._nodes:
                if (n['name'] == name):
                    n['params'] = nd.getParams()
                    ok = True
                    break
        return ok


    def training(self, lr: float, inTns: numpy.ndarray, outTns: numpy.ndarray,
                 trgTns: numpy.ndarray, outAccurate : []) -> bool:
        """Training net - cycle fwd<->bwd with calc error."""

        if (not(self._net) and not(self._createNet())):
            return False

        insz = snLSize()
        insz.bsz = inTns.shape[0]
        insz.ch = inTns.shape[1]
        insz.h = inTns.shape[2]
        insz.w = inTns.shape[3]
        indata = inTns.__array_interface__['data'][0]

        outsz = snLSize()
        outsz.bsz = outTns.shape[0]
        outsz.ch = outTns.shape[1]
        outsz.h = outTns.shape[2]
        outsz.w = outTns.shape[3]
        outdata = outTns.__array_interface__['data'][0]

        trgsz = snLSize()
        trgsz.bsz = trgTns.shape[0]
        trgsz.ch = trgTns.shape[1]
        trgsz.h = trgTns.shape[2]
        trgsz.w = trgTns.shape[3]
        trgdata = trgTns.__array_interface__['data'][0]

        pfun = _LIB.snTraining
        pfun.restype = ctypes.c_bool
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_float, ctypes.Structure, ctypes.POINTER(ctypes.c_float),
                         ctypes.Structure, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))

        cAccurate = ctypes.c_float(0)

        ok = pfun(self._net, ctypes.c_float(lr), insz, snFloat_p(indata), outsz,
                  snFloat_p(outdata), snFloat_p(trgdata), snFloat_p(ctypes.addressof(cAccurate)))

        outAccurate[0] = cAccurate.value

        return ok


    def forward(self, isLern : bool, inTns : numpy.ndarray, outTns : numpy.ndarray) -> bool:
        """Forward net."""

        if (not(self._net) and not(self._createNet())):
            return False

        insz = snLSize()
        insz.bsz = inTns.shape[0]
        insz.ch = inTns.shape[1]
        insz.h = inTns.shape[2]
        insz.w = inTns.shape[3]
        indata = inTns.__array_interface__['data'][0]

        outsz = snLSize()
        outsz.bsz = outTns.shape[0]
        outsz.ch = outTns.shape[1]
        outsz.h = outTns.shape[2]
        outsz.w = outTns.shape[3]
        outdata = outTns.__array_interface__['data'][0]

        pfun = _LIB.snForward
        pfun.restype = ctypes.c_bool
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.Structure, ctypes.POINTER(ctypes.c_float),
                         ctypes.Structure, ctypes.POINTER(ctypes.c_float))

        return pfun(self._net, isLern, insz, indata, outsz, outdata)


    def backward(self, lr : float, gradTns : numpy.ndarray) -> bool:
        """Backward net."""

        if (not(self._net) and not(self._createNet())):
            return False

        insz = snLSize()
        insz.bsz = gradTns.shape[0]
        insz.ch = gradTns.shape[1]
        insz.h = gradTns.shape[2]
        insz.w = gradTns.shape[3]
        indata = gradTns.__array_interface__['data'][0]

        pfun = _LIB.snBackward
        pfun.restype = ctypes.c_bool
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_float, ctypes.Structure, ctypes.POINTER(ctypes.c_float))

        return pfun(self._net, lr, insz, indata)


    def loadAllWeightFromFile(self, weightPath : str) -> bool:
        """load All Weight From File"""

        if (not (self._net) and not (self._createNet())):
            return False

        pfun = _LIB.snLoadAllWeightFromFile
        pfun.restype = ctypes.c_bool
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_char_p)

        return pfun(self._net, c_str(weightPath))


    def saveAllWeightToFile(self, weightPath: str) -> bool:
        """save All Weight to File"""

        if (not (self._net) and not (self._createNet())):
            return False

        pfun = _LIB.snSaveAllWeightToFile
        pfun.restype = ctypes.c_bool
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_char_p)

        return pfun(self._net, c_str(weightPath))


    def _createNet(self) -> bool:
        """Create net."""

        if (self._net): return True

        nsz = len(self._nodes)
        if (nsz == 0): return False

        beginNode = self._nodes[0]['NodeName']
        prevEndNode = self._nodes[nsz - 1]['NodeName']

        for i in range(0, nsz):
            if (self._nodes[i]['OperatorName'] == 'Input'):
                beginNode = self._nodes[i]['NextNodes']
            if (self._nodes[i]['NextNodes'] == 'Output'):
                prevEndNode = self._nodes[i]['NodeName']
                self._nodes[i]['NextNodes'] = "EndNet"

        for i in range(0, nsz):
            if (self._nodes[i]['OperatorName'] == 'Input'):
                self._nodes.pop(i)
                break

        ss = {
            'BeginNet':{
                'NextNodes' : beginNode
            },
            'Nodes' : self._nodes,
            'EndNet': {
                'PrevNode': prevEndNode
            }
        }

        return self._createNetJn(json.dumps(ss))


    def _createNetJn(self, jnNet : str) -> bool:
        """Create net."""

        if (self._net): return True

        err = ctypes.create_string_buffer(256)

        errCBack = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)

        pfun = _LIB.snCreateNet
        pfun.restype = ctypes.c_void_p
        pfun.argtypes = (ctypes.c_char_p, ctypes.c_char_p, errCBack, ctypes.c_void_p)

        self._errCBack = errCBack(lambda mess, obj : print('SNet ' + str(mess)))

        self._net = pfun(c_str(jnNet), err, self._errCBack, 0)

        if (not self._net):
            print(err)

        return self._net