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

#include "stdafx.h"

/// adaptive gradient method
void opt_adagrad(SN_Base::snFloat* dW, SN_Base::snFloat* ioWGr, SN_Base::snFloat* ioW, size_t sz, SN_Base::snFloat alpha = 0.001F, SN_Base::snFloat lambda = 0.F, SN_Base::snFloat eps = 1e-8F);

/// RMSprop
void opt_RMSprop(SN_Base::snFloat* dW, SN_Base::snFloat* ioWGr, SN_Base::snFloat* ioW, size_t sz, SN_Base::snFloat alpha = 0.001F, SN_Base::snFloat lambda = 0.F, SN_Base::snFloat mu = 0.9F, SN_Base::snFloat eps = 1e-8F);

/// adam
void opt_adam(SN_Base::snFloat* dW, SN_Base::snFloat* iodWPrev, SN_Base::snFloat* ioWGr, SN_Base::snFloat* ioW, size_t sz, SN_Base::snFloat alpha = 0.001F, SN_Base::snFloat lambda = 0.F, SN_Base::snFloat muWd = 0.9F, SN_Base::snFloat muGr = 0.9F, SN_Base::snFloat eps = 1e-8F);

/// SGD
void opt_sgd(SN_Base::snFloat* dW, SN_Base::snFloat* ioW, size_t sz, SN_Base::snFloat alpha = 0.001F, SN_Base::snFloat lambda = 0.F);

/// SGD with momentum
void opt_sgdMoment(SN_Base::snFloat* dW, SN_Base::snFloat* iodWPrev, SN_Base::snFloat* ioW, size_t sz, SN_Base::snFloat alpha = 0.01F, SN_Base::snFloat lambda = 0.F, SN_Base::snFloat mu = 0.9F);
