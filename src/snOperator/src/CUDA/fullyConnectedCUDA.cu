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
#include "snOperator/src/Operator/fullyConnected.h"

using namespace std;
using namespace SN_Base;
         
#ifndef cuCHECK
#define cuCHECK(func) if (func != 0){ ERROR_MESS("CUDA error: " + cudaGetErrorString(cudaGetLastError())); return;}
#endif


struct gpuParams{

    cublasHandle_t cuBLAS = 0;

    snFloat* d_in = 0;
    snFloat* d_w = 0;
    snFloat* d_dw = 0;
    snFloat* d_out = 0;
    snFloat* d_grout = 0;

    size_t inszMem = 0;
};

void FullyConnected::iniParamCUDA(const snSize& insz, size_t kernel, void** pGpuPrm){
 
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
        
        if (!gpuClearMem_){
            cuCHECK(cudaMalloc(&gpuPrm->d_in, bsz * ida * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm->d_w, (ida + 1) * kernel * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm->d_out, bsz * kernel * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm->d_grout, bsz * ida * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm->d_dw, (ida + 1) * kernel * sizeof(snFloat)));
        }
    }
    else if (!gpuClearMem_ && (gpuPrm->inszMem < ida * bsz)){
          
        cuCHECK(cudaFree(gpuPrm->d_in));    gpuPrm->d_in = 0;
        cuCHECK(cudaFree(gpuPrm->d_w));     gpuPrm->d_w = 0;
        cuCHECK(cudaFree(gpuPrm->d_out));   gpuPrm->d_out = 0;
        cuCHECK(cudaFree(gpuPrm->d_grout)); gpuPrm->d_grout = 0;
        cuCHECK(cudaFree(gpuPrm->d_dw));    gpuPrm->d_dw = 0;

        cuCHECK(cudaMalloc(&gpuPrm->d_in, bsz * ida * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_w, (ida + 1) * kernel * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, bsz * kernel * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, bsz * ida * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_dw, (ida + 1) * kernel * sizeof(snFloat)));

        gpuPrm->inszMem = ida * bsz;
    }
}
         
void FullyConnected::freeParamCUDA(void* gpuPrms){
   
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;

    if (!gpuPrm) return;
       
    cublasDestroy(gpuPrm->cuBLAS);
    cuCHECK(cudaFree(gpuPrm->d_in));
    cuCHECK(cudaFree(gpuPrm->d_out));
    cuCHECK(cudaFree(gpuPrm->d_w));
    cuCHECK(cudaFree(gpuPrm->d_dw));
    cuCHECK(cudaFree(gpuPrm->d_grout));
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

void FullyConnected::forwardCUDA(size_t kernel, const snSize& insz, snFloat* input, snFloat* weight, snFloat* output, void* gpuPrms){
   
    cudaSetDevice(gpuDeviceId_);
       
    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    int ida = int(insz.w * insz.h * insz.d), bsz = int(insz.n), krn = int(kernel);
       
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_in, bsz * ida * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_w, (ida + 1) * kernel * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, bsz * kernel * sizeof(snFloat)));
    }

    cuCHECK(cublasSetMatrix(bsz, ida, sizeof(snFloat), input, bsz, gpuPrm->d_in, bsz));
  
    cuCHECK(cublasSetMatrix(ida, krn, sizeof(snFloat), weight, ida, gpuPrm->d_w, ida));
   
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
        gpuPrm->d_w,                   // W
        krn,                           // W, step to next W (W21 - W11)
        gpuPrm->d_in,                  // In
        ida,                           // In, step to next X (X21 - X11)  
        &beta,                         // β
        gpuPrm->d_out,                 // Out
        krn));                         // Out, step to next Y (Y21 - Y11) 
    
    // +bias
    cuFwdBias <<< int(insz.n), 128 >>> (kernel, insz, gpuPrm->d_w, gpuPrm->d_out);

    // result
    cuCHECK(cublasGetMatrix(bsz, krn, sizeof(snFloat), gpuPrm->d_out, bsz, output, bsz));
  
    if (gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_in));   gpuPrm->d_in = 0;
        cuCHECK(cudaFree(gpuPrm->d_w));    gpuPrm->d_w = 0;
        cuCHECK(cudaFree(gpuPrm->d_out));  gpuPrm->d_out = 0;
    }
}

__global__ void cuBwdBias(size_t kernel, snSize insz, snFloat* gradIn, snFloat* dWOut){
    
    dWOut += insz.w * insz.h * insz.d * kernel;
    unsigned int k = threadIdx.x;
    while (k < kernel){
   
        snFloat* grin = gradIn + k, b = 0;
        for (size_t j = 0; j < insz.n; ++j)
            b += grin[kernel * j];

        dWOut[k] = b;
        k += blockDim.x;
    }
}

void FullyConnected::backwardCUDA_GW(size_t kernel, snFloat* weight,
    const snSize& insz, snFloat* input, snFloat* gradIn, snFloat* gradOut, snFloat* dWOut, void* gpuPrms){
   
    cudaSetDevice(gpuDeviceId_);
        
    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    int ida = int(insz.w * insz.h * insz.d), bsz = int(insz.n), krn = int(kernel);
           
    snFloat* d_grin = gpuPrm->d_out;
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_in, bsz * ida * sizeof(snFloat)));          
        cuCHECK(cudaMalloc(&gpuPrm->d_w, (ida + 1) * kernel * sizeof(snFloat)));        
        cuCHECK(cudaMalloc(&d_grin, bsz * kernel * sizeof(snFloat)));    
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, bsz * ida * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_dw, (ida + 1) * kernel * sizeof(snFloat)));
    }

    cuCHECK(cublasSetMatrix(bsz, ida, sizeof(snFloat), input, bsz, gpuPrm->d_in, bsz));
  
    cuCHECK(cublasSetMatrix(bsz, krn, sizeof(snFloat), gradIn, bsz, d_grin, bsz));

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
        d_grin,                  // GrIn
        krn,                     // GrIn, step to next 
        gpuPrm->d_in,            // In
        ida,                     // In, step to next  X (X21 - X11)  
        &beta,                   // β               
        gpuPrm->d_dw,            // dW            
        krn));                   // dW, step to next 
 
    // bias
    cuBwdBias <<< 1, 128 >>> (kernel, insz, d_grin, gpuPrm->d_dw);
  
    // result
    cuCHECK(cudaMemcpy(dWOut, gpuPrm->d_dw, (ida + 1) * krn * sizeof(snFloat), cudaMemcpyDeviceToHost));
 

    cuCHECK(cublasSetMatrix(ida, krn, sizeof(snFloat), weight, ida, gpuPrm->d_w, ida));

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
        gpuPrm->d_w,             // W
        krn,                     // W, step to next 
        d_grin,                  // GrIn
        krn,                     // GrIn, step to next 
        &beta,                   // β               
        gpuPrm->d_grout,         // GrOut                                  
        ida));                   // GrOut, step to next 
     
   
    // result
    cuCHECK(cublasGetMatrix(bsz, ida, sizeof(snFloat), gpuPrm->d_grout, bsz, gradOut, bsz));
 
    if (gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_in));     gpuPrm->d_in = 0;
        cuCHECK(cudaFree(gpuPrm->d_w));      gpuPrm->d_w = 0;
        cuCHECK(cudaFree(d_grin));           gpuPrm->d_out = 0;
        cuCHECK(cudaFree(gpuPrm->d_grout));  gpuPrm->d_grout = 0;
        cuCHECK(cudaFree(gpuPrm->d_dw));     gpuPrm->d_dw = 0;
    }
}

void FullyConnected::backwardCUDA_G(size_t kernel, snFloat* weight, const snSize& insz, snFloat* gradIn, snFloat* gradOut, void* gpuPrms){
  
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    int ida = int(insz.w * insz.h * insz.d), bsz = int(insz.n), krn = int(kernel);

    snFloat* d_grin = gpuPrm->d_out;
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_w, ida * kernel * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&d_grin, bsz * kernel * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, bsz * ida * sizeof(snFloat)));
    }

    cuCHECK(cublasSetMatrix(bsz, krn, sizeof(snFloat), gradIn, bsz, d_grin, bsz));

    cuCHECK(cublasSetMatrix(ida, krn, sizeof(snFloat), weight, ida, gpuPrm->d_w, ida));

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
        gpuPrm->d_w,             // W
        krn,                     // W, step to next 
        d_grin,                  // GrIn
        krn,                     // GrIn, step to next 
        &beta,                   // β         
        gpuPrm->d_grout,         // GrOut                          
        ida));                   // GrOut, step to next 

    cuCHECK(cublasGetMatrix(bsz, ida, sizeof(snFloat), gpuPrm->d_grout, bsz, gradOut, bsz));

    if (gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_w));      gpuPrm->d_w = 0;
        cuCHECK(cudaFree(d_grin));           gpuPrm->d_out = 0;
        cuCHECK(cudaFree(gpuPrm->d_grout));  gpuPrm->d_grout = 0;
    }
}

#endif 