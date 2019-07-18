//
// SkyNet Project
// Copyright (C) 2018 by Contributors <https://github.com/Tyill/skynet>
//
// This code is licensed under the MIT License.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
#pragma once

// инициализация весов

#include "snBase/snBase.h"
#include "structurs.h"

void weightInit(SN_Base::snFloat* ioW, size_t sz, size_t fan_in, size_t fan_out, weightInitType wtype);

void wi_uniform(SN_Base::snFloat* ioW, size_t sz);

// 2010 Xavier Glorot
void wi_xavier(SN_Base::snFloat* ioW, size_t sz, size_t fan_in, size_t fan_out);

// 1998 Yann LeCun
void wi_lecun(SN_Base::snFloat* ioW, size_t sz, size_t fan_out);

// 2015 Kaiming He
void wi_he(SN_Base::snFloat* ioW, size_t sz, size_t fan_in);
