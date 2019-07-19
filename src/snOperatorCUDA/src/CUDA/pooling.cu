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

#include <cuda_runtime.h>
#include <cudnn.h>
#include "../stdafx.h"
#include "snOperatorCUDA/src/Operator/pooling.h"

using namespace std;
using namespace SN_Base;


struct gpuParams{

    cudnnHandle_t cudnn = 0;
    cudnnPoolingDescriptor_t pool_desc = 0;
    cudnnTensorDescriptor_t in_desc = 0;
    cudnnTensorDescriptor_t out_desc = 0;
    cudnnTensorDescriptor_t grin_desc = 0;
    cudnnTensorDescriptor_t grout_desc = 0;    
   
};

void Pooling::iniParamCUDA(bool isLern, const snSize& insz, const snSize& outsz, const poolParams& poolPrms, void** pGpuPrm){
     
    bool isFirst = false;

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
       
        cudnnHandle_t cudnn = nullptr;
        cuCHECK(cudnnCreate(&cudnn));
        gpuPrm->cudnn = cudnn;              

        isFirst = true;
    }

    // input
    cudnnTensorDescriptor_t in_desc = nullptr;
    cuCHECK(cudnnCreateTensorDescriptor(&in_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, int(insz.n), int(insz.d), int(insz.h), int(insz.w)));
    if (!isFirst)
        cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->in_desc));
    gpuPrm->in_desc = in_desc;
     
    // pool
    cudnnPoolingDescriptor_t pool_desc = nullptr;
    cuCHECK(cudnnCreatePoolingDescriptor(&pool_desc));

    cudnnPoolingMode_t poolT = cudnnPoolingMode_t::CUDNN_POOLING_MAX;
    if (poolPrms.type == poolType::avg)
        poolT = cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
   
    cuCHECK(cudnnSetPooling2dDescriptor(pool_desc, poolT, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        int(poolPrms.kernel), int(poolPrms.kernel), 0, 0, int(poolPrms.stride), int(poolPrms.stride)));
    if (!isFirst)
        cuCHECK(cudnnDestroyPoolingDescriptor(gpuPrm->pool_desc));
    gpuPrm->pool_desc = pool_desc;

    // output
    int out_n = 0, out_c = 0, out_h = 0, out_w = 0;
    cuCHECK(cudnnGetPooling2dForwardOutputDim(pool_desc, in_desc,
        &out_n, &out_c, &out_h, &out_w));

    if (outsz != snSize(out_w, out_h, out_c, out_n)){
        ERROR_MESS("CUDA error: outsz != snSize(out_w, out_h, out_c, out_n)");
        return;
    }

    cudnnTensorDescriptor_t out_desc;
    cuCHECK(cudnnCreateTensorDescriptor(&out_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));
    if (!isFirst)
        cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->out_desc));
    gpuPrm->out_desc = out_desc;

    if (isLern){
        // grout
        cudnnTensorDescriptor_t grout_desc;
        cuCHECK(cudnnCreateTensorDescriptor(&grout_desc));
        cuCHECK(cudnnSetTensor4dDescriptor(grout_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, int(insz.n), int(insz.d), int(insz.h), int(insz.w)));
        if (!isFirst)
            cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->grout_desc));
        gpuPrm->grout_desc = grout_desc;

        // grin
        cudnnTensorDescriptor_t grin_desc;
        cuCHECK(cudnnCreateTensorDescriptor(&grin_desc));
        cuCHECK(cudnnSetTensor4dDescriptor(grin_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            out_n, out_c, out_h, out_w));
        if (!isFirst)
            cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->grin_desc));
        gpuPrm->grin_desc = grin_desc;
    }

}

void Pooling::freeParamCUDA(void* gpuPrms){
    
    gpuParams* gpuPrm = (gpuParams*)gpuPrms;

    if (!gpuPrm) return;
    
    cuCHECK(cudnnDestroy(gpuPrm->cudnn));
    cuCHECK(cudnnDestroyPoolingDescriptor(gpuPrm->pool_desc));
    cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->in_desc));
    cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->out_desc));
      
}

__global__ void cuFiltrNegative(snSize outsz, snFloat* out){

    out += blockIdx.x * outsz.w * outsz.h * outsz.d;
       
    unsigned int k = threadIdx.x;
    while (k < outsz.d){

        snFloat* pOut = out + outsz.w * outsz.h * k;
        for (size_t j = 0; j < (outsz.w * outsz.h); ++j)
            if (pOut[j] < 0) pOut[j] = 0.0;

        k += blockDim.x;
    }    
}

void Pooling::forwardCUDA(const poolParams& poolPrms, const snSize& insz, const snFloat* input,
    const snSize& outsz, snFloat* output, void* gpuPrms){
    
    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
      
    // run
    snFloat alpha = 1.f, beta = 0.f;
    cuCHECK(cudnnPoolingForward(gpuPrm->cudnn,
        gpuPrm->pool_desc,
        &alpha,
        gpuPrm->in_desc,
        input,
        &beta,
        gpuPrm->out_desc,
        output));
   
    // filtrNegative
    dim3 dimBlock(256);
    dim3 dimGrid(int(outsz.n));

    cuFiltrNegative << < dimGrid, dimBlock >> >(outsz, output);
  
}

void Pooling::backwardCUDA(const poolParams& poolPrms, const snSize& outsz, const snFloat* output, const snFloat* gradIn,
    const snSize& insz, const snFloat* input, snFloat* gradOut, void* gpuPrms){
       
    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    
    // run
    snFloat alpha = 1.f, beta = 0.f;
    cuCHECK(cudnnPoolingBackward(gpuPrm->cudnn,
        gpuPrm->pool_desc,
        &alpha,
        gpuPrm->out_desc,
        output,
        gpuPrm->grin_desc,
        gradIn,
        gpuPrm->in_desc,
        input,
        &beta,
        gpuPrm->grout_desc,
        gradOut));
     
}
