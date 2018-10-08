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
#include "Lib/OpenBLAS/cblas.h"
#include "snOperator/src/Operator/fullyConnected.h"
#include <omp.h>  

using namespace std;
using namespace SN_Base;


void FullyConnected::forwardCPU(size_t kernel, const snSize& insz, snFloat* input, snFloat* weight, snFloat* output){

    size_t imSz = insz.w * insz.h * insz.d;

    // Out = α * In * W + βC
    // In - data input matrix - values from the previous layer
    // W - weights matrix
    // Out - data output matrix
    cblas_sgemm(CBLAS_ORDER::CblasRowMajor,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_TRANSPOSE::CblasNoTrans,
        blasint(insz.n),                       // In, rows
        blasint(kernel),                       // W, cols
        blasint(imSz),                         // In, cols, W, rows              
        1.0F,                                  // α
        input,                                 // In
        blasint(imSz),                         // In, step to next In
        weight,                                // W
        blasint(kernel),                       // W, step to next W (W21 - W11) 
        0.0,                                   // β
        output,                                // Out
        blasint(kernel));                      // Out, step to next Out (Y21 - Y11) 
       
    // +bias
    weight += imSz * kernel;               
    for (size_t i = 0; i < insz.n; ++i){
   
        snFloat* out = output + kernel * i;
        for (size_t j = 0; j < kernel; ++j)
            out[j] += weight[j];            
    }
}

void FullyConnected::backwardCPU_GW(size_t kernel, snFloat* weight,
    const snSize& insz, snFloat* input, snFloat* gradIn, snFloat* gradOut, snFloat* dWOut){

    size_t imSz = insz.w * insz.h * insz.d;

    // Grad by weight
    // dW = αIn^T * GrIn + βdW
    // In - data input matrix - values from the previous layer
    // GrIn - gradient matrix from the next layer
    cblas_sgemm(CBLAS_ORDER::CblasRowMajor,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_TRANSPOSE::CblasNoTrans,
        blasint(imSz),                 // In, rows     
        blasint(kernel),               // GrIn, cols
        blasint(insz.n),               // In, cols. GrIn, rows
        1.0F / insz.n,                 // α
        input,                         // In
        blasint(imSz),                 // In, step to next
        gradIn,                        // GrIn
        blasint(kernel),               // GrIn, step to next
        0.0F,                          // β
        dWOut,                         // dW
        blasint(kernel));              // dW, step to next

    // Gradient for previous layer
    // GrOut = αGrIn * W^T + βGrOut
    // GrIn - gradient matrix from the next layer
    // W - weight
    cblas_sgemm(CBLAS_ORDER::CblasRowMajor,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_TRANSPOSE::CblasTrans,
        blasint(insz.n),               // GrIn, rows
        blasint(imSz),                 // W, cols
        blasint(kernel),               // GrIn, cols. W, rows
        1.0F,                          // α
        gradIn,                        // GrIn
        blasint(kernel),               // GrIn, step to next (I21 - I11) 
        weight,                        // W
        blasint(kernel),               // W, step to next W (W21 - W11) 
        0.0F,                          // β
        gradOut,                       // GrOut
        blasint(imSz));                // GrOut, step to next Y (Y21 - Y11) 

    // bias
    dWOut += imSz * kernel;
    for (size_t i = 0; i < kernel; ++i){

        snFloat* grin = gradIn + i, b = 0;
        for (size_t j = 0; j < insz.n; ++j)
            b += grin[kernel * j];

        dWOut[i] = b;
    }
}

void FullyConnected::backwardCPU_G(size_t kernel, snFloat* weight, const snSize& insz, snFloat* gradIn, snFloat* gradOut){

    size_t imSz = insz.w * insz.h * insz.d;

    // Gradient for previous layer
    // GrOut = αGrIn * W^T + βGrOut
    // GrIn - gradient matrix from the next layer
    // W - weight
    cblas_sgemm(CBLAS_ORDER::CblasRowMajor,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_TRANSPOSE::CblasTrans,
        blasint(insz.n),               // GrIn, rows
        blasint(imSz),                 // W, cols
        blasint(kernel),               // GrIn, cols
        1.0F,                          // α
        gradIn,                        // GrIn
        blasint(kernel),               // GrIn, step to next (I21 - I11) 
        weight,                        // W
        blasint(kernel),               // W,  step to next W (W21 - W11) 
        0.0F,                          // β 
        gradOut,                       // GrOut
        blasint(imSz));                // GrOut, step to next Y (Y21 - Y11) 
}


#ifndef SN_CUDA

void FullyConnected::iniParamCUDA(const snSize& insz, size_t kernel, std::map<std::string, void*>& auxPrm){
    ERROR_MESS("CUDA non compiler");
}

void FullyConnected::freeParamCUDA(std::map<std::string, void*>& auxPrm){
    ERROR_MESS("CUDA non compiler");
}

void FullyConnected::forwardCUDA(size_t kernel, const snSize& insz, snFloat* input, snFloat* weight, snFloat* output, std::map<std::string, void*>& auxPrm){
    ERROR_MESS("CUDA non compiler");
}

void FullyConnected::backwardCUDA_GW(size_t kernel, snFloat* weight,
    const snSize& insz, snFloat* input, snFloat* gradIn, snFloat* gradOut, snFloat* dWOut, std::map<std::string, void*>&){
    ERROR_MESS("CUDA non compiler");
}

void FullyConnected::backwardCUDA_G(size_t kernel, snFloat* weight, const snSize& insz, snFloat* gradIn, snFloat* gradOut, std::map<std::string, void*>&){
    ERROR_MESS("CUDA non compiler");
}


#endif


#ifndef SN_OpenCL

void FullyConnected::iniParamOCL(const SN_Base::snSize& insz, size_t kernel, void** gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void FullyConnected::freeParamOCL(void* auxPrm){
    ERROR_MESS("OpenCL non compiler");
}

void FullyConnected::forwardOCL(size_t kernel, const snSize& insz, snFloat* input, snFloat* weight, snFloat* output, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void FullyConnected::backwardOCL_GW(size_t kernel, snFloat* weight,
    const snSize& insz, snFloat* input, snFloat* gradIn, snFloat* gradOut, snFloat* dWOut, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void FullyConnected::backwardOCL_G(size_t kernel, snFloat* weight, const snSize& insz, snFloat* gradIn, snFloat* gradOut, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}


#endif