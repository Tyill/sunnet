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


__global__ void softMaxACrossEntropyFwd(snSize iosz, snFloat* inout){
      
    size_t inStepByD = iosz.w * iosz.h,     // step out by input
           inStepByN = inStepByD * iosz.d;  // step out by batch       

    // gridDim.x - number of out layers
 
    inout += blockIdx.x * inStepByN;
           
    __shared__ int tmax;
    __shared__ snFloat tsumm;

    tmax = 0;
    tsumm = 0;

    __syncthreads();

    unsigned int i = threadIdx.x;
    while (i < inStepByN){

        atomicMax(&tmax, int(inout[i] * 100.F));
       
        i += blockDim.x;
    }

    __syncthreads();
    
    i = threadIdx.x;
    while (i < inStepByN){
       
        inout[i] = ((inout[i] - tmax / 100.F) > -20) ? exp(inout[i] - tmax / 100.F) : 0.1E-8F;

        atomicAdd(&tsumm, inout[i]);
             
        i += blockDim.x;
    }

    __syncthreads();

    i = threadIdx.x;
    while (i < inStepByN){

        inout[i] /= tsumm;

        i += blockDim.x;
    }   
}

__global__ void softMaxACrossEntropyBwd(snSize iosz, snFloat* out, snFloat* targ, snFloat* grad){

    size_t inStepByD = iosz.w * iosz.h,     // step out by input
           inStepByN = inStepByD * iosz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size  
    
    grad += blockIdx.x * inStepByD + blockIdx.y * inStepByN;
    out += blockIdx.x * inStepByD + blockIdx.y * inStepByN;
    targ += blockIdx.x * inStepByD + blockIdx.y * inStepByN;

    unsigned int i = threadIdx.x;

    while (i < inStepByD){

        grad[i] = out[i] - targ[i];

        i += blockDim.x;
    } 
}

__global__ void binaryCrossEntropyBwd(snSize iosz, snFloat* out, snFloat* targ, snFloat* grad){

    size_t inStepByD = iosz.w * iosz.h,     // step out by input
           inStepByN = inStepByD * iosz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size  

    grad += blockIdx.x * inStepByD + blockIdx.y * inStepByN;
    out += blockIdx.x * inStepByD + blockIdx.y * inStepByN;
    targ += blockIdx.x * inStepByD + blockIdx.y * inStepByN;

    unsigned int i = threadIdx.x;

    while (i < inStepByD){
        
        grad[i] = (out[i] - targ[i]) / (out[i] * (1.F - out[i]));

        i += blockDim.x;
    }
}

__global__ void regressionMSEBwd(snSize iosz, snFloat* out, snFloat* targ, snFloat* grad){

    size_t inStepByD = iosz.w * iosz.h,     // step out by input
        inStepByN = inStepByD * iosz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size  

    grad += blockIdx.x * inStepByD + blockIdx.y * inStepByN;
    out += blockIdx.x * inStepByD + blockIdx.y * inStepByN;
    targ += blockIdx.x * inStepByD + blockIdx.y * inStepByN;

    unsigned int i = threadIdx.x;

    while (i < inStepByD){
        
        grad[i] = 2 * (out[i] - targ[i]) / inStepByN;

        i += blockDim.x;
    }
}


void lossForward(const snSize& sz, snFloat* inout, lossType loss){

    dim3 dimBlock(256);
    dim3 dimGrid(int(sz.n));

    switch (loss){
        case lossType::softMaxACrossEntropy:
            softMaxACrossEntropyFwd <<<dimGrid, dimBlock >>>(sz, inout);
            break;

        case lossType::binaryCrossEntropy:
            break;

        case lossType::regressionMSE: 
            break;
    }
}

void lossBackward(const snSize& sz, snFloat* out, snFloat* targ, snFloat* grad, lossType loss){

    dim3 dimBlock(128);
    dim3 dimGrid(int(sz.d), int(sz.n));

    switch (loss){
      case lossType::softMaxACrossEntropy:
          
          softMaxACrossEntropyBwd << <dimGrid, dimBlock >> >(sz, out, targ, grad); 
          break;    
      
      case lossType::binaryCrossEntropy:
      
          binaryCrossEntropyBwd << <dimGrid, dimBlock >> >(sz, out, targ, grad);
          break;
                                           
      case lossType::regressionMSE: // Mean Square Error
      
          regressionMSEBwd << <dimGrid, dimBlock >> >(sz, out, targ, grad);
          break;
    }   
}
