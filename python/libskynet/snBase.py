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
import numpy

if sys.version_info[0] > 2:
    py_str = lambda x: x.encode('utf-8')
else:
    py_str = lambda x: x

# type definitions
snFloat = ctypes.c_float
snFloat_p = ctypes.POINTER(snFloat)
snHandle = ctypes.c_void_p

def c_str(string : str) -> ctypes.c_char_p:
    """Create ctypes char * from a Python string."""

    return ctypes.c_char_p(py_str(string))

class snLSize(ctypes.Structure):
     _fields_ = [('w', ctypes.c_size_t),
                ('h', ctypes.c_size_t),
                ('ch', ctypes.c_size_t),
                ('bsz', ctypes.c_size_t)]

def snArrayFloat(values):
    """Create ctypes array from a Python array."""

    out = (snFloat * len(values))()
    out[:] = values
    return out