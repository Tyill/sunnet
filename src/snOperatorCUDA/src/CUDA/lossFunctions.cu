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
#include "../stdafx.h"
#include "snOperatorCUDA/src/Operator/lossFunction.h"

using namespace std;
using namespace SN_Base;


__global__ void softMaxACrossEntropy(snSize iosz, snFloat* inout){
      
    size_t inStepByD = iosz.w * iosz.h,     // step out by input
           inStepByN = inStepByD * iosz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    inout += blockIdx.x * inStepByN;
           
    __shared__ int tmax;
    __shared__ snFloat tsumm;

    tmax = 0;
    tsumm = 0;

    __syncthreads();

    unsigned int i = threadIdx.x;

    while (i < inStepByN){

        atomicMax(&tmax, int(inout[i]));
       
        i += blockDim.x;
    }

    __syncthreads();
    
    while (i < inStepByN){
       
        inout[i] = (inout[i] - tmax > -20) ? exp(inout[i] - tmax) : 0.1E-8F;

        atomicAdd(&tsumm, inout[i]);
             
        i += blockDim.x;
    }

    __syncthreads();

    while (i < inStepByN){

        inout[i] /= tsumm;

        i += blockDim.x;
    }   
}

void lossForward(const snSize& insz, snFloat* inout, lossType loss){

    dim3 dimBlock(256);
    dim3 dimGrid(int(insz.n));

    switch (loss){
        case lossType::softMaxACrossEntropy:
            softMaxACrossEntropy <<<dimGrid, dimBlock >>>(insz, inout);
            break;
    }
}

void lossBackward(const Tensor& inTns, snFloat* out, snFloat* targ, snFloat* grad, lossType loss){

    
   
}
