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
    _userCBack = {}

    def __init__(self, jnNet : str = '', weightPath : str = ''):
        """
        init
        :param jnNet: architec of net json
        :param weightPath: weight file path
        """
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
            pfun(self._net)


    def getErrorStr(self) -> str:
        """
        last error string
        :return: '' ok
        """
        if (not self._net):
            return 'net not create'

        pfun = _LIB.snGetLastErrorStr
        pfun.restype = None
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_char_p)

        err = ctypes.create_string_buffer(256)

        self._net = pfun(self._net, err)

        return err


    def addNode(self, name : str, nd : snOperator, nextNodes : str):
        """
        add Node
        :param name: name node
        :param nd: tensor node
        :param nextNodes: next nodes through a space
        :return: True ok
        """
        self._nodes.append({'NodeName': name, 'OperatorName':  nd.name(), 'OperatorParams': nd.getParams().copy(), 'NextNodes': nextNodes})

        return self


    def updateNode(self, name : str, nd : snOperator) -> bool:
        """
        Update params node
        :param name: name node
        :param nd: tensor node
        :return: True ok
        """

        ok = False
        if (self._net):
            ok = _LIB.snSetParamNode(self._net, c_str(name),  c_str(nd.getParams()))
        else:
            for n in self._nodes:
                if (n['name'] == name):
                    n['params'] = nd.getParams().copy()
                    ok = True
                    break
        return ok


    def training(self, lr: float, inTns: numpy.ndarray, outTns: numpy.ndarray,
                 trgTns: numpy.ndarray, outAccurate : []) -> bool:
        """
        Training net - cycle fwd<->bwd with calc error
        :param lr: lerning rate
        :param inTns: in tensor
        :param outTns: out tensor
        :param trgTns: target tensor
        :param outAccurate: accurate
        :return: True ok
        """

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
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_float, snLSize, ctypes.POINTER(ctypes.c_float),
                         snLSize, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))

        cAccurate = ctypes.c_float(0)

        ok = pfun(self._net, ctypes.c_float(lr), insz, snFloat_p(indata), outsz,
                  snFloat_p(outdata), snFloat_p(trgdata), snFloat_p(ctypes.addressof(cAccurate)))

        outAccurate[0] = cAccurate.value

        return ok


    def forward(self, isLern : bool, inTns : numpy.ndarray, outTns : numpy.ndarray) -> bool:
        """
        Forward action
        :param isLern: is lerning?
        :param inTns: in tensor
        :param outTns: out tensor
        :return: True ok
        """

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
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_bool, snLSize, ctypes.POINTER(ctypes.c_float),
                         snLSize, ctypes.POINTER(ctypes.c_float))

        return pfun(self._net, isLern, insz, snFloat_p(indata), outsz, snFloat_p(outdata))


    def backward(self, lr : float, gradTns : numpy.ndarray) -> bool:
        """
        Backward action
        :param lr: lerning rate
        :param gradTns: in gradient error tensor
        :return: True ok
        """

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
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_float, snLSize, ctypes.POINTER(ctypes.c_float))

        return pfun(self._net, ctypes.c_float(lr), insz, snFloat_p(indata))


    def addUserCallBack(self, ucbName: str, ucb) -> bool:
        """
        User callback for 'UserCBack' layer and 'LossFunction' layer
        :param ucbName: cback name
        :param ucb: cback function
        :return: True ok

        ucb = function(None,
                       str,               # name user cback
                       str,               # name node
                       bool,              # current action forward(true) or backward(false)
                       inLayer: ndarray,  # input layer - receive from prev node
                       outLayer: ndarray, # output layer - send to next node
                       )
        """

        if (not (self._net) and not (self._createNet())):
            return False

        def c_ucb(ucbName: ctypes.c_char_p,                                # name user cback
                  nodeName: ctypes.c_char_p,                               # name node
                  isFwdBwd: ctypes.c_bool,                                 # current action forward(true) or backward(false)
                  inSize: snLSize,                                         # input layer size - receive from prev node
                  inData: ctypes.POINTER(ctypes.c_float),                  # input layer - receive from prev node
                  outSize: ctypes.POINTER(snLSize),                        # output layer size - send to next node
                  outData: ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), # output layer - send to next node
                  auxUData: ctypes.c_void_p):                              # aux used data

            insz = inSize.w * inSize.h * inSize.ch * inSize.bsz

            dbuffer = (ctypes.c_float * insz).from_address(ctypes.addressof(inData.contents))
            inLayer = numpy.frombuffer(dbuffer, ctypes.c_float).reshape((inSize.bsz, inSize.ch, inSize.h, inSize.w))

            outLayer = inLayer.copy()

            ucb(ucbName.decode("utf-8"), nodeName.decode("utf-8"), isFwdBwd, inLayer, outLayer)

            outSize.contents.bsz = outLayer.shape[0]
            outSize.contents.ch = outLayer.shape[1]
            outSize.contents.h = outLayer.shape[2]
            outSize.contents.w = outLayer.shape[3]

            outsz = outLayer.shape[0] * outLayer.shape[1] * outLayer.shape[2] * outLayer.shape[3]

            cbuff = self._userCBack[ucbName.decode("utf-8")][1]
            if (self._userCBack[ucbName.decode("utf-8")][2] != outsz):
                self._userCBack[ucbName.decode("utf-8")][2] = outsz
                cbuff = self._userCBack[ucbName.decode("utf-8")][1] = (ctypes.c_float * outsz)()

            addrBuff = ctypes.cast(ctypes.addressof(cbuff), ctypes.POINTER(ctypes.c_float))
            ctypes.memmove(ctypes.addressof(outData.contents), ctypes.addressof(addrBuff),
                           ctypes.sizeof(ctypes.POINTER(ctypes.c_float)))

            ctypes.memmove(ctypes.addressof(cbuff), snFloat_p(outLayer.__array_interface__['data'][0]),
                           ctypes.sizeof(ctypes.c_float) * outsz)


        self._userCBack[ucbName] = [snUserCBack(c_ucb), (ctypes.c_float)(), 0]

        pfun = _LIB.snAddUserCallBack
        pfun.restype = ctypes.c_bool
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_char_p, snUserCBack, ctypes.c_void_p)

        return pfun(self._net, c_str(ucbName), self._userCBack[ucbName][0], 0)


    def getWeightNode(self, nodeName: str, weight: [numpy.ndarray]) -> bool:
        """
         get Weight of Node
        :param nodeName: node name
        :param weight: out array weight as list[0]
        :return: True ok
        """

        if (not (self._net) and not (self._createNet())):
            return False

        pfun = _LIB.snGetWeightNode
        pfun.restype = ctypes.c_bool
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_char_p,
           ctypes.POINTER(snLSize), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))

        wsize = snLSize()
        wdata = ctypes.POINTER(ctypes.c_float)()

        if (pfun(self._net, c_str(nodeName), ctypes.pointer(wsize), ctypes.byref(wdata))):

            wsz = wsize.w * wsize.h * wsize.ch * wsize.bsz

            dbuffer = (ctypes.c_float * wsz).from_address(ctypes.addressof(wdata.contents))
            weight[0] = numpy.frombuffer(dbuffer, ctypes.c_float).copy().reshape((wsize.bsz, wsize.ch, wsize.h, wsize.w))

            pfun = _LIB.snFreeResources
            pfun.restype = None
            pfun.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_char_p)

            pfun(wdata, ctypes.c_char_p(0))

            return True
        else:
            return False


    def setWeightNode(self, nodeName: str, weight: numpy.ndarray) -> bool:
        """
        set weight of node
        :param nodeName: node name
        :param weight: set array weight
        :return: True ok
        """

        if (not (self._net) and not (self._createNet())):
            return False

        wsize = snLSize()
        wsize.bsz = weight.shape[0]
        wsize.ch = weight.shape[1]
        wsize.h = weight.shape[2]
        wsize.w = weight.shape[3]
        inw = weight.__array_interface__['data'][0]

        pfun = _LIB.snSetWeightNode
        pfun.restype = ctypes.c_bool
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_char_p,
                         snLSize, ctypes.POINTER(ctypes.c_float))

        return pfun(self._net, c_str(nodeName), wsize, snFloat_p(inw))


    def loadAllWeightFromFile(self, weightPath : str) -> bool:
        """
        load All Weight From File
        :param weightPath: weight Path file
        :return: True ok
        """

        if (not (self._net) and not (self._createNet())):
            return False

        pfun = _LIB.snLoadAllWeightFromFile
        pfun.restype = ctypes.c_bool
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_char_p)

        return pfun(self._net, c_str(weightPath))


    def saveAllWeightToFile(self, weightPath: str) -> bool:
        """
        save All Weight to File
        :param weightPath: weight Path file
        :return: True ok
        """

        if (not (self._net) and not (self._createNet())):
            return False

        pfun = _LIB.snSaveAllWeightToFile
        pfun.restype = ctypes.c_bool
        pfun.argtypes = (ctypes.c_void_p, ctypes.c_char_p)

        return pfun(self._net, c_str(weightPath))


    def getGetArchitecNet(self) -> str:
        """
        architecture of net
        :return: arch in json. '' - error
        """

        if (not (self._net) and not (self._createNet())):
            return False

        pfun = _LIB.snGetArchitecNet
        pfun.restype = ctypes.c_bool
        pfun.argtypes = (ctypes.c_void_p,  ctypes.POINTER(ctypes.c_char_p))

        ss = ctypes.c_char_p()

        if (pfun(self._net, ctypes.byref(ss))):

            ret = ss.value.decode("utf-8")

            pfun = _LIB.snFreeResources
            pfun.restype = None
            pfun.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_char_p)

            pfun(ctypes.cast(0, ctypes.POINTER(ctypes.c_float)), ss)

            return ret
        else:
            return ''


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

        pfun = _LIB.snCreateNet
        pfun.restype = ctypes.c_void_p
        pfun.argtypes = (ctypes.c_char_p, ctypes.c_char_p, snErrCBack, ctypes.c_void_p)

        self._errCBack = snErrCBack(lambda mess, obj : print('SNet ' + str(mess)))

        err = ctypes.create_string_buffer(256)
        self._net = pfun(c_str(jnNet), err, self._errCBack, 0)

        if (not self._net):
            print(err)

        return self._net

