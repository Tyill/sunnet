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
#include "SNOperator/src/Operator/convolution.h"

using namespace std;
using namespace SN_Base;
          

void Convolution::iniParamCUDA(snSize insz, size_t kernel, map<string, snFloat*>& gpuPrm){

}

void Convolution::freeParamCUDA(map<std::string, snFloat*>& gpuPrm){

    for (auto p : gpuPrm)
        cudaFree(p.second);
}

void Convolution::forwardCUDA(size_t kernel, size_t fWidth, size_t fHeight, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* output, map<string, snFloat*>& auxPrm){

 
}

void Convolution::backwardCUDA_GW(size_t kernel, size_t fWidth, size_t fHeight, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, map<string, snFloat*>&){


}

void Convolution::backwardCUDA_G(size_t kernel, size_t fWidth, size_t fHeight, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* gradIn, snFloat* gradOut, map<string, snFloat*>&){


}

#endif 
