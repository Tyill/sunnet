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

#include "stdafx.h"

using namespace std;
using namespace SN_Base;


void paddingOffs(bool in2out, size_t paddW, size_t paddH, const snSize& insz, snFloat* in, snFloat* out){

    /// copy with offset padding for each image    
    size_t sz = insz.h * insz.d * insz.n, stW = insz.w, stH = insz.h;
    if (in2out){
        in += (stW + paddW * 2) * paddH;
        for (size_t i = 0; i < sz; ++i){

            if ((i % stH == 0) && (i > 0))
                in += (stW + paddW * 2) * paddH * 2;

            in += paddW;
            for (size_t j = 0; j < stW; ++j)
                out[j] = in[j];
            in += paddW + stW;

            out += stW;
        }
    }
    else{
        in += (stW + paddW * 2) * paddH;
        for (size_t i = 0; i < sz; ++i){

            if ((i % stH == 0) && (i > 0))
                in += (stW + paddW * 2) * paddH * 2;

            in += paddW;
            for (size_t j = 0; j < stW; ++j)
                in[j] = out[j];
            in += paddW + stW;

            out += stW;
        }
    }
}
