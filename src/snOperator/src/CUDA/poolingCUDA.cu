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

#ifdef SN_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../stdafx.h"
#include "SNOperator/src/Operator/pooling.h"

using namespace std;
using namespace SN_Base;
          

void Pooling::iniParamCUDA(snSize insz, size_t kernel, map<string, snFloat*>& auxPrm){

   
}

void Pooling::freeParamCUDA(map<string, snFloat*>& gpuPrm){
    
    for (auto p : gpuPrm)
        cudaFree(p.second);
}

void Pooling::forwardCUDA(int type, size_t kernel, snSize insz, snFloat* input,
    snSize outsz, snFloat* output, size_t* outputInx, map<string, snFloat*>& auxPrm){

   
}

void Pooling::backwardCUDA(int type, size_t kernel, snSize outsz, size_t* outputInx, snFloat* gradIn,
    snSize insz, snFloat* gradOut, map<string, snFloat*>& auxPrm){

 
}




#endif 