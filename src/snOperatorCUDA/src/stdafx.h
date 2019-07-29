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

#include <algorithm>
#include <omp.h>
#include "snBase/snBase.h"

#define PROFILE_START double ctm = omp_get_wtime(); 
#define PROFILE_END(func) g_statusMess(this, name_ + " " + node_ + " " + func + " " + std::to_string(omp_get_wtime() - ctm)); ctm = omp_get_wtime(); 

#define ERROR_MESS(mess) g_statusMess(this, name_ + " '" + node_ + "' error: " + mess);

void g_statusMess(SN_Base::OperatorBase* opr, const std::string& mess);

void g_userCBack(SN_Base::OperatorBase* opr, const std::string& cbname, const std::string& node,
    bool fwBw, const SN_Base::snSize& insz, SN_Base::snFloat* in, SN_Base::snSize& outsz, SN_Base::snFloat** out);

#define cuCHECK(func) if (func != 0){ ERROR_MESS("CUDA error: " + cudaGetErrorString(cudaGetLastError())); return;}

#define cuAssert(func) ASSERT_MESS(func == 0, std::string("CUDA error: ") + cudaGetErrorString(cudaGetLastError()));                                                  
