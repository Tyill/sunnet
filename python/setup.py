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

from __future__ import absolute_import
import os
import sys

from setuptools import find_packages

kwargs = {}
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension
    kwargs = {'install_requires': ['numpy<=1.15.2,>=1.8.2', 'requests<2.19.0,>=2.18.4', 'graphviz<0.9.0,>=0.8.1'], 'zip_safe': False}

with open("README.md", "r") as fh:
    long_description = fh.read()

CURRENT_DIR = os.path.normpath(os.path.dirname(__file__))

if os.name == 'nt':
    trgPath = 'lib/site-packages/libskynet'
else:
    trgPath = 'lib/python3.5/dist-packages/libskynet'

dll_path = os.listdir(os.path.join(CURRENT_DIR, 'lib'))
lib_path = [os.path.join(CURRENT_DIR, 'lib') + os.path.sep + p for p in dll_path]

__version__ = '1.0.1'

setup(
    name="libskynet",
    version=__version__,
    url="https://github.com/Tyill/skynet",
    packages=find_packages(),
    description="neural net with blackjack and hookers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    data_files=[(trgPath, lib_path)],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    **kwargs)