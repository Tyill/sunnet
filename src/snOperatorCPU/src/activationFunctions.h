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

#include "snBase/snBase.h"
#include "structurs.h"

// fv - value, df - deriv

void activationForward(size_t sz, SN_Base::snFloat* data, activeType);

void activationBackward(size_t sz, SN_Base::snFloat* data, activeType);

void fv_sigmoid(SN_Base::snFloat* ioVal, size_t sz);
void df_sigmoid(SN_Base::snFloat* inSigm, size_t sz);

void fv_relu(SN_Base::snFloat* ioVal, size_t sz);
void df_relu(SN_Base::snFloat* inRelu, size_t sz);

void fv_leakyRelu(SN_Base::snFloat* ioVal, size_t sz, SN_Base::snFloat minv = 0.01F);
void df_leakyRelu(SN_Base::snFloat* inRelu, size_t sz, SN_Base::snFloat minv = 0.01F);

void fv_elu(SN_Base::snFloat* ioVal, size_t sz, SN_Base::snFloat minv = 0.01F);
void df_elu(SN_Base::snFloat* inElu, size_t sz, SN_Base::snFloat minv = 0.01F);
