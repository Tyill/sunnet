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

import sys
import ctypes

# type definitions
snFloat_p = lambda x : ctypes.cast(x, ctypes.POINTER(ctypes.c_float))

def c_str(string : str) -> ctypes.c_char_p:
    """Create ctypes char * from a Python string."""

    if sys.version_info[0] > 2:
        py_str = lambda x: x.encode('utf-8')
    else:
        py_str = lambda x: x

    return ctypes.c_char_p(py_str(string))

class snLSize(ctypes.Structure):
     _fields_ = [('w', ctypes.c_size_t),
                 ('h', ctypes.c_size_t),
                 ('ch', ctypes.c_size_t),
                 ('bsz', ctypes.c_size_t)]

class snBNorm(ctypes.Structure):
     _fields_ = [('mean', ctypes.POINTER(ctypes.c_float)),
                 ('varce', ctypes.POINTER(ctypes.c_float)),
                 ('scale', ctypes.POINTER(ctypes.c_float)),
                 ('schift', ctypes.POINTER(ctypes.c_float))]

snErrCBack = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)

snUserCBack = ctypes.CFUNCTYPE(None,
                               ctypes.c_char_p,                                # name user cback
                               ctypes.c_char_p,                                # name node
                               ctypes.c_bool,                                  # current action forward(true) or backward(false)
                               snLSize,                                        # input layer size - receive from prev node
                               ctypes.POINTER(ctypes.c_float),                 # input layer - receive from prev node
                               ctypes.POINTER(snLSize),                        # output layer size - send to next node
                               ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), # output layer - send to next node
                               ctypes.c_void_p                                 # aux used data
                               )