#!/usr/bin/env python

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

# coding: utf-8

from __future__ import absolute_import
import os
import ctypes


from . import snNet
from . import snType
from . import snOperator

libname = 'libskynet.so'
if os.name == 'nt':
    libname = 'libskynet.dll'

libname = os.path.abspath(
    os.path.join(os.path.dirname(__file__), libname))

_LIB = ctypes.CDLL(libname)


def _snVersionLib() -> str:
    """
    version library
    :return: version
    """

    pfun = _LIB.snVersionLib
    pfun.restype = None
    pfun.argtypes = (ctypes.c_char_p,)

    ver = ctypes.create_string_buffer(32)
    pfun(ver)

    return ver.value.decode("utf-8")

# current version
__version__ = _snVersionLib()