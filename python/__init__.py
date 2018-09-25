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
import os
import platform
import logging
import ctypes

def _load_lib():
    """Load library by searching possible path."""
    lib_path = find_lib_path()

    if (lib_path == ''):
        raise RuntimeError('Cannot find the library libskynet')

    lib = ctypes.CDLL(lib_path)

    return lib

def find_lib_path():

    nm = 'libskynet.so'
    if os.name == 'nt':
        nm = 'libskynet.dll'

    for p in sys.path:
        for f in os.listdir(p):
            if (f == nm):
                return os.path.join(p, f)

    return ''


_LIB = _load_lib()

__all__ = ["snType", "snNet", "snOperator"]

__version__ = "1.0.0.1"

