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

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../stdafx.h"
#include "snOperatorCUDA/src/Operator/fullyConnected.h"

using namespace std;
using namespace SN_Base;
        
struct gpuParams{

    cublasHandle_t cuBLAS = 0;
     
};

void FullyConnected::iniParamCUDA(bool isLern, const snSize& insz, size_t kernel, void** pGpuPrm){
 
    cudaSetDevice(gpuDeviceId_);

    size_t ida = insz.w * insz.h * insz.d, bsz = insz.n;

    gpuParams* gpuPrm = (gpuParams*)*pGpuPrm;
    if (!gpuPrm){
    
        cudaDeviceProp cu_deviceProps;
        cudaGetDeviceProperties(&cu_deviceProps, 0);
        if (cu_deviceProps.major < 3){
            ERROR_MESS("%s requires SM >= 3.0");
            return;
        }
        gpuPrm = new gpuParams();
        memset(gpuPrm, 0, sizeof(gpuParams));
        *pGpuPrm = gpuPrm;

        cublasHandle_t cuHandle = nullptr;
        cuCHECK(cublasCreate(&cuHandle));

        gpuPrm->cuBLAS = cuHandle;
    }
}
         
void FullyConnected::freeParamCUDA(void* gpuPrms){
       
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;

    if (!gpuPrm) return;
       
    cublasDestroy(gpuPrm->cuBLAS);
}

__global__ void cuFwdBias(size_t kernel, snSize insz, snFloat* weight, snFloat* output){
       
    weight += insz.w * insz.h * insz.d * kernel;
   
    snFloat* out = output + kernel * blockIdx.x;
    unsigned int k = threadIdx.x;
    while (k < kernel){

        out[k] += weight[k];

        k += blockDim.x;
    }   
}

void FullyConnected::forwardCUDA(size_t kernel, const snSize& insz, const snFloat* input, const snFloat* weight, snFloat* output, void* gpuPrms){
        
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    int ida = int(insz.w * insz.h * insz.d), bsz = int(insz.n), krn = int(kernel);
   
  //  cuCHECK(cublasSetMatrix(bsz, ida, sizeof(snFloat), input, bsz, gpuPrm->d_in, bsz));
  
   // cuCHECK(cudaMemcpy(gpuPrm->d_w, weight, (ida + 1) * krn * sizeof(snFloat), cudaMemcpyHostToDevice));

    // Out = α * W * In + βC
    // In - data input matrix - values from the previous layer
    // W - weights matrix
    // Out - output matrix
    float alpha = 1.0f, beta = 0.0f;
    cuCHECK(cublasSgemm(gpuPrm->cuBLAS,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        krn,                           // W, cols
        bsz,                           // In, rows
        ida,                           // In, cols, В М - rows            
        &alpha,                        // α
        weight,                        // W
        krn,                           // W, step to next W (W21 - W11)
        input,                         // In
        ida,                           // In, step to next X (X21 - X11)  
        &beta,                         // β
        output,                        // Out
        krn));                         // Out, step to next Y (Y21 - Y11) 
    
    // +bias
    cuFwdBias <<< int(insz.n), 128 >>> (kernel, insz, gpuPrm->d_w, gpuPrm->d_out);
    
}

__global__ void cuBwdBias(size_t kernel, snSize insz, snFloat* gradIn, snFloat* dWOut){
    
    dWOut += insz.w * insz.h * insz.d * kernel;
    unsigned int k = threadIdx.x;
    while (k < kernel){
   
        snFloat* grin = gradIn + k, b = 0;
        for (size_t j = 0; j < insz.n; ++j)
            b += grin[kernel * j];

        dWOut[k] = b / insz.n;
        k += blockDim.x;
    }
}

void FullyConnected::backwardCUDA_GW(size_t kernel, const snFloat* weight,
    const snSize& insz, const snFloat* input, const snFloat* gradIn, snFloat* gradOut, snFloat* dWOut, void* gpuPrms){
       
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    int ida = int(insz.w * insz.h * insz.d), bsz = int(insz.n), krn = int(kernel);
   /*
    cuCHECK(cublasSetMatrix(bsz, ida, sizeof(snFloat), input, bsz, gpuPrm->d_in, bsz));
  
    cuCHECK(cublasSetMatrix(bsz, krn, sizeof(snFloat), gradIn, bsz, d_grin, bsz));
*/

    // Weight gradient
    // dW = αIn^T * GrIn + βdW
    // In - data input matrix from previous layer
    // GrIn - gradient matrix from the next layer
    float alpha = 1.0F / insz.n, beta = 0.0f;
    cuCHECK(cublasSgemm(gpuPrm->cuBLAS,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        krn,                     // GrIn, cols
        ida,                     // In, cols (+1 - X0)      
        bsz,                     // In, rows
        &alpha,                  // α                
        gradIn,                  // GrIn
        krn,                     // GrIn, step to next 
        input,                   // In
        ida,                     // In, step to next  X (X21 - X11)  
        &beta,                   // β               
        dWOut,                   // dW            
        krn));                   // dW, step to next 
 
    // bias
    cuBwdBias <<< 1, 128 >>> (kernel, insz, d_grin, gpuPrm->d_dw);
     
//    cuCHECK(cublasSetMatrix(ida, krn, sizeof(snFloat), weight, ida, gpuPrm->d_w, ida));

    //// Gradient for previous layer
    //// GrOut = αGrIn * W^T + βGrOut
    //// GrIn - gradient matrix from the next layer
    //// W - weight
    alpha = 1.F;
    cuCHECK(cublasSgemm(gpuPrm->cuBLAS,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        ida,                     // W, cols     
        bsz,                     // W, rows
        krn,                     // GrIn, cols
        &alpha,                  // α                               
        weight,                  // W
        krn,                     // W, step to next 
        gradIn,                  // GrIn
        krn,                     // GrIn, step to next 
        &beta,                   // β               
        gradOut,                 // GrOut                                  
        ida));                   // GrOut, step to next 
    
}

void FullyConnected::backwardCUDA_G(size_t kernel, const snFloat* weight, const snSize& insz, const snFloat* gradIn, snFloat* gradOut, void* gpuPrms){
      
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    int ida = int(insz.w * insz.h * insz.d), bsz = int(insz.n), krn = int(kernel);
        
    /*cuCHECK(cublasSetMatrix(bsz, krn, sizeof(snFloat), gradIn, bsz, d_grin, bsz));

    cuCHECK(cublasSetMatrix(ida, krn, sizeof(snFloat), weight, ida, gpuPrm->d_w, ida));
*/

    //// Gradient for previous layer
    //// GrOut = αGrIn * W^T + βGrOut
    //// GrIn - gradient matrix from the next layer
    //// W - weight
    float alpha = 1.0F, beta = 0.0f;
    cuCHECK(cublasSgemm(gpuPrm->cuBLAS,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        ida,                     // W, cols
        bsz,                     // W, rows
        krn,                     // GrIn, cols
        &alpha,                  // α                               
        weight,                  // W
        krn,                     // W, step to next 
        gradIn,                  // GrIn
        krn,                     // GrIn, step to next 
        &beta,                   // β         
        gradOut,                 // GrOut                          
        ida));                   // GrOut, step to next 
        
}