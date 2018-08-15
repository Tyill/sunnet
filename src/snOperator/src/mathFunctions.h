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

struct batchNormParam{
    SN_Base::snFloat* norm;      ///< нормирован вх значения
    SN_Base::snFloat* mean;      ///< среднее вх значений
    SN_Base::snFloat* varce;     ///< дисперсия вх значений
    SN_Base::snFloat* scale;     ///< коэф γ
    SN_Base::snFloat* dScale;    ///< dγ
    SN_Base::snFloat* schift;    ///< коэф β
    SN_Base::snFloat* dSchift;   ///< dβ
    SN_Base::snFloat* onc;       ///< 1й вектор
    SN_Base::snFloat lr = 0.001F; ///< коэф для изменения γ и β
};

void fwdBatchNorm(SN_Base::snSize insz,
                  SN_Base::snFloat* in,
                  SN_Base::snFloat* out,
                  batchNormParam);

void bwdBatchNorm(SN_Base::snSize insz, 
                  SN_Base::snFloat* gradIn,
                  SN_Base::snFloat* gradOut,
                  batchNormParam);
