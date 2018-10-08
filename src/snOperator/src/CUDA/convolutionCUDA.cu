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

#ifdef SN_CUDNN

#include <cuda_runtime.h>
#include <cudnn.h>
#include "../stdafx.h"
#include "SNOperator/src/Operator/convolution.h"

using namespace std;
using namespace SN_Base;

#ifndef cuCHECK
#define cuCHECK(func) if (func != 0){ ERROR_MESS("CUDA error: " + cudaGetErrorString(cudaGetLastError())); return;}
#endif

struct gpuParams{

    cudnnHandle_t cudnn = 0;
    cudnnConvolutionDescriptor_t conv_desc = 0;
    cudnnTensorDescriptor_t in_desc = 0;
    cudnnTensorDescriptor_t out_desc = 0;
    cudnnTensorDescriptor_t grin_desc = 0;
    cudnnTensorDescriptor_t grout_desc = 0;
    cudnnFilterDescriptor_t w_desc = 0;
    cudnnFilterDescriptor_t dw_desc = 0;
    cudnnTensorDescriptor_t bias_desc = 0;

    cudnnConvolutionFwdAlgo_t algoFwd;
    cudnnConvolutionBwdDataAlgo_t algoBwdData;
    cudnnConvolutionBwdFilterAlgo_t algoBwdW;

    size_t wsFwdSz = 0;
    size_t wsBwdDataSz = 0;
    size_t wsBwdWSz = 0;

    snFloat* d_in = 0;
    snFloat* d_w = 0;
    snFloat* d_dw = 0;
    snFloat* d_bias = 0;
    snFloat* d_out = 0;
    snFloat* d_grout = 0;
    void* d_wsFwd = 0;
    void* d_wsBwdData = 0;
    void* d_wsBwdW = 0;
};

void Convolution::iniParamCUDA(const snSize& insz, const snSize& outsz, 
    const convParams& prms, void** pGpuPrm){

    cudaSetDevice(gpuDeviceId_);
    
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
        cuCHECK(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm->in_desc));
    gpuPrm->in_desc = in_desc;
    
    // grout
    cudnnTensorDescriptor_t grout_desc;
    cuCHECK(cudnnCreateTensorDescriptor(&grout_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(grout_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, int(insz.n), int(insz.d), int(insz.h), int(insz.w)));
    if (!isFirst)
        cuCHECK(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm->grout_desc));
    gpuPrm->grout_desc = grout_desc;

    // w      
    cudnnFilterDescriptor_t w_desc = nullptr;
    cuCHECK(cudnnCreateFilterDescriptor(&w_desc));
    cuCHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        int(outsz.d), int(insz.d), int(prms.fHeight), int(prms.fWidth)));
    if (!isFirst)
        cuCHECK(cudnnDestroyFilterDescriptor((cudnnFilterDescriptor_t)gpuPrm->w_desc));
    gpuPrm->w_desc = w_desc;

    // dw     
    cudnnFilterDescriptor_t dw_desc = nullptr;
    cuCHECK(cudnnCreateFilterDescriptor(&dw_desc));
    cuCHECK(cudnnSetFilter4dDescriptor(dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        int(outsz.d), int(insz.d), int(prms.fHeight), int(prms.fWidth)));
    if (!isFirst)
        cuCHECK(cudnnDestroyFilterDescriptor((cudnnFilterDescriptor_t)gpuPrm->dw_desc));
    gpuPrm->dw_desc = dw_desc;

    // conv
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cuCHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    cuCHECK(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, int(prms.stride), int(prms.stride), int(prms.dilate), int(prms.dilate),
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    if (!isFirst)
        cuCHECK(cudnnDestroyConvolutionDescriptor((cudnnConvolutionDescriptor_t)gpuPrm->conv_desc));
    gpuPrm->conv_desc = conv_desc;

    // output
    int out_n = 0, out_c = 0, out_h = 0, out_w = 0;
    cuCHECK(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, w_desc,
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
        cuCHECK(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm->out_desc));
    gpuPrm->out_desc = out_desc;

    cudnnTensorDescriptor_t grin_desc;
    cuCHECK(cudnnCreateTensorDescriptor(&grin_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(grin_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));
    if (!isFirst)
        cuCHECK(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm->grin_desc));
    gpuPrm->grin_desc = grin_desc;
         
    // bias
    cudnnTensorDescriptor_t bias_desc;
    cuCHECK(cudnnCreateTensorDescriptor(&bias_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        1, out_c, 1, 1));
    if (!isFirst)
        cuCHECK(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm->bias_desc));
    gpuPrm->bias_desc = bias_desc;

    // algorithm
    cudnnConvolutionFwdAlgo_t algoFwd;
    cuCHECK(cudnnGetConvolutionForwardAlgorithm(gpuPrm->cudnn, in_desc, w_desc, conv_desc, out_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algoFwd));    
    gpuPrm->algoFwd = algoFwd;

    cudnnConvolutionBwdDataAlgo_t algoBwdData;
    cuCHECK(cudnnGetConvolutionBackwardDataAlgorithm(gpuPrm->cudnn, w_desc, grin_desc, conv_desc, grout_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algoBwdData));    
    gpuPrm->algoBwdData = algoBwdData;

    cudnnConvolutionBwdFilterAlgo_t algoBwdW;
    cuCHECK(cudnnGetConvolutionBackwardFilterAlgorithm(gpuPrm->cudnn, in_desc, grin_desc, conv_desc, dw_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algoBwdW));    
    gpuPrm->algoBwdW = algoBwdW;

    // workspace
    size_t wsFwdSz = 0;
    cuCHECK(cudnnGetConvolutionForwardWorkspaceSize(gpuPrm->cudnn, in_desc, w_desc, conv_desc, out_desc, algoFwd, &wsFwdSz));
    gpuPrm->wsFwdSz = wsFwdSz;

    size_t wsBwdDataSz = 0;
    cuCHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(gpuPrm->cudnn, w_desc, grin_desc, conv_desc, grout_desc, algoBwdData, &wsBwdDataSz));
    gpuPrm->wsBwdDataSz = wsBwdDataSz;

    size_t wsBwdWSz = 0;
    cuCHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(gpuPrm->cudnn, in_desc, grin_desc, conv_desc, dw_desc, algoBwdW, &wsBwdWSz));
    gpuPrm->wsBwdWSz = wsBwdWSz;

    if (isFirst && !gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_w, prms.fWidth * prms.fHeight * insz.d * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_dw, prms.fWidth * prms.fHeight * insz.d * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_wsFwd, wsFwdSz));
        cuCHECK(cudaMalloc(&gpuPrm->d_wsBwdData, wsBwdDataSz));
        cuCHECK(cudaMalloc(&gpuPrm->d_wsBwdW, wsBwdWSz));
        cuCHECK(cudaMalloc(&gpuPrm->d_bias, outsz.d * sizeof(snFloat)));
    }   
    else if (!gpuClearMem_){        
        cuCHECK(cudaFree(gpuPrm->d_in));        gpuPrm->d_in = 0; 
        cuCHECK(cudaFree(gpuPrm->d_w));         gpuPrm->d_w = 0;  
        cuCHECK(cudaFree(gpuPrm->d_dw));        gpuPrm->d_dw = 0;
        cuCHECK(cudaFree(gpuPrm->d_out));       gpuPrm->d_out = 0;
        cuCHECK(cudaFree(gpuPrm->d_grout));     gpuPrm->d_grout = 0;
        cuCHECK(cudaFree(gpuPrm->d_wsFwd));     gpuPrm->d_wsFwd = 0;
        cuCHECK(cudaFree(gpuPrm->d_wsBwdData)); gpuPrm->d_wsBwdData = 0;
        cuCHECK(cudaFree(gpuPrm->d_wsBwdW));    gpuPrm->d_wsBwdW = 0;
        cuCHECK(cudaFree(gpuPrm->d_bias));      gpuPrm->d_bias = 0;

        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_w, prms.fWidth * prms.fHeight * insz.d * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_dw, prms.fWidth * prms.fHeight * insz.d * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_wsFwd, wsFwdSz));
        cuCHECK(cudaMalloc(&gpuPrm->d_wsBwdData, wsBwdDataSz));
        cuCHECK(cudaMalloc(&gpuPrm->d_wsBwdW, wsBwdWSz));
        cuCHECK(cudaMalloc(&gpuPrm->d_bias, outsz.d * sizeof(snFloat)));
    }
}

void Convolution::freeParamCUDA(void* gpuPrms){
  
    cudaSetDevice(gpuDeviceId_);
   
    gpuParams* gpuPrm = (gpuParams*)gpuPrms;

    if (!gpuPrm) return;
       
    cuCHECK(cudnnDestroy(gpuPrm->cudnn));
    cuCHECK(cudnnDestroyConvolutionDescriptor(gpuPrm->conv_desc));     
    cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->in_desc));
    cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->out_desc));
    cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->grin_desc));
    cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->grout_desc));
    cuCHECK(cudnnDestroyFilterDescriptor(gpuPrm->w_desc));
    cuCHECK(cudnnDestroyFilterDescriptor(gpuPrm->dw_desc));
    cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->bias_desc));
        
    cuCHECK(cudaFree(gpuPrm->d_in));
    cuCHECK(cudaFree(gpuPrm->d_w));
    cuCHECK(cudaFree(gpuPrm->d_dw));
    cuCHECK(cudaFree(gpuPrm->d_bias));
    cuCHECK(cudaFree(gpuPrm->d_out));
    cuCHECK(cudaFree(gpuPrm->d_grout));
    cuCHECK(cudaFree(gpuPrm->d_wsFwd));
    cuCHECK(cudaFree(gpuPrm->d_wsBwdData));
    cuCHECK(cudaFree(gpuPrm->d_wsBwdW));
}

__global__ void cuFwdBias(snSize outsz, snFloat* bias, snFloat* output){

    size_t osz = outsz.w * outsz.h;
      
    snFloat* pOut = output + osz * outsz.d * blockIdx.x;
    unsigned int d = threadIdx.x;
    while (d < outsz.d){
                
        snFloat b = bias[d];
        for (size_t j = 0; j < osz; ++j)
            pOut[j] += b;

        pOut += osz * blockDim.x;
       
        d += blockDim.x;
    }    
}

void Convolution::forwardCUDA(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, void* gpuPrms){

    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_w, prms.fWidth * prms.fHeight * insz.d * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_bias, outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_wsFwd, gpuPrm->wsFwdSz));
    }

    // input
    cuCHECK(cudaMemcpy(gpuPrm->d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // weight
    size_t wsz = outsz.d * insz.d * prms.fHeight * prms.fWidth;
    cuCHECK(cudaMemcpy(gpuPrm->d_w, weight, wsz * sizeof(snFloat), cudaMemcpyHostToDevice));
    cuCHECK(cudaMemcpy(gpuPrm->d_bias, weight + wsz, outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));
    
    // run
    snFloat alpha = 1.f, beta = 0.f;   
    cuCHECK(cudnnConvolutionForward(gpuPrm->cudnn,
        &alpha,
        gpuPrm->in_desc,
        gpuPrm->d_in,
        gpuPrm->w_desc,
        gpuPrm->d_w,
        gpuPrm->conv_desc,
        gpuPrm->algoFwd,
        gpuPrm->d_wsFwd,
        gpuPrm->wsFwdSz,
        &beta,
        gpuPrm->out_desc,
        gpuPrm->d_out));
 
    // +bias
    cuFwdBias <<< int(insz.n), 128 >>> (outsz, gpuPrm->d_bias, gpuPrm->d_out);
 
    // result
    cuCHECK(cudaMemcpy(output, gpuPrm->d_out, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_in));     gpuPrm->d_in = 0;
        cuCHECK(cudaFree(gpuPrm->d_w));      gpuPrm->d_w = 0;
        cuCHECK(cudaFree(gpuPrm->d_bias));   gpuPrm->d_bias = 0;
        cuCHECK(cudaFree(gpuPrm->d_wsFwd));  gpuPrm->d_wsFwd = 0;
        cuCHECK(cudaFree(gpuPrm->d_out));    gpuPrm->d_out = 0;
    }
}

void Convolution::backwardCUDA_GW(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, void* gpuPrms){
   
    cudaSetDevice(gpuDeviceId_);
       
    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    void* d_grin = gpuPrm->d_out;
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_w, prms.fWidth * prms.fHeight * insz.d * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_dw, prms.fWidth * prms.fHeight * insz.d * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_bias, outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&d_grin, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_wsBwdData, gpuPrm->wsBwdDataSz));
        cuCHECK(cudaMalloc(&gpuPrm->d_wsBwdW, gpuPrm->wsBwdWSz));
    }

    // input
    cuCHECK(cudaMemcpy(gpuPrm->d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // grin
    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // weight
    size_t wsz = outsz.d * insz.d * prms.fHeight * prms.fWidth;
    cuCHECK(cudaMemcpy(gpuPrm->d_w, weight, wsz * sizeof(snFloat), cudaMemcpyHostToDevice));
    cuCHECK(cudaMemcpy(gpuPrm->d_bias, weight + wsz, outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));

    // run       
    snFloat alpha = 1.f, beta = 0.f;
    cuCHECK(cudnnConvolutionBackwardData(gpuPrm->cudnn,
        &alpha,
        gpuPrm->w_desc,
        gpuPrm->d_w,
        gpuPrm->grin_desc,
        d_grin,
        gpuPrm->conv_desc,
        gpuPrm->algoBwdData,
        gpuPrm->d_wsBwdData,
        gpuPrm->wsBwdDataSz,
        &beta,
        gpuPrm->grout_desc,
        gpuPrm->d_grout));

    cuCHECK(cudnnConvolutionBackwardFilter(gpuPrm->cudnn,
        &alpha,
        gpuPrm->in_desc,
        gpuPrm->d_in,
        gpuPrm->grin_desc,
        d_grin,
        gpuPrm->conv_desc,
        gpuPrm->algoBwdW,
        gpuPrm->d_wsBwdW,
        gpuPrm->wsBwdWSz,
        &beta,
        gpuPrm->dw_desc,
        gpuPrm->d_dw));
    
    cuCHECK(cudnnConvolutionBackwardBias(gpuPrm->cudnn,
        &alpha,
        gpuPrm->grin_desc,
        d_grin,
        &beta,
        gpuPrm->bias_desc,
        gpuPrm->d_bias));

    // результ
    cuCHECK(cudaMemcpy(gradOut, gpuPrm->d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
    cuCHECK(cudaMemcpy(dWeightOut, gpuPrm->d_dw, wsz * sizeof(snFloat), cudaMemcpyDeviceToHost));
    cuCHECK(cudaMemcpy(dWeightOut + wsz, gpuPrm->d_bias, outsz.d * sizeof(snFloat), cudaMemcpyDeviceToHost));
   
    if (gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_in));         gpuPrm->d_in = 0;
        cuCHECK(cudaFree(gpuPrm->d_w));          gpuPrm->d_w = 0;
        cuCHECK(cudaFree(d_grin));               gpuPrm->d_out = 0;
        cuCHECK(cudaFree(gpuPrm->d_grout));      gpuPrm->d_grout = 0;
        cuCHECK(cudaFree(gpuPrm->d_dw));         gpuPrm->d_dw = 0;
        cuCHECK(cudaFree(gpuPrm->d_bias));       gpuPrm->d_bias = 0;
        cuCHECK(cudaFree(gpuPrm->d_wsBwdData));  gpuPrm->d_wsBwdData = 0;
        cuCHECK(cudaFree(gpuPrm->d_wsBwdW));     gpuPrm->d_wsBwdW = 0;
    }
}

void Convolution::backwardCUDA_G(const convParams& prms,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, void* gpuPrms){
    
    cudaSetDevice(gpuDeviceId_);
      
   
    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    void* d_grin = gpuPrm->d_out;
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_w, prms.fWidth * prms.fHeight * insz.d * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&d_grin, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_wsBwdData, gpuPrm->wsBwdDataSz));      
    }
  
    // grin
    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // weight
    size_t wsz = outsz.d * insz.d * prms.fHeight * prms.fWidth;
    cuCHECK(cudaMemcpy(gpuPrm->d_w, weight, wsz * sizeof(snFloat), cudaMemcpyHostToDevice));
  
    // run      
    snFloat alpha = 1.f, beta = 0.f;
    cuCHECK(cudnnConvolutionBackwardData(gpuPrm->cudnn,
        &alpha,
        gpuPrm->w_desc,
        gpuPrm->d_w,
        gpuPrm->grin_desc,
        d_grin,
        gpuPrm->conv_desc,
        gpuPrm->algoBwdData,
        gpuPrm->d_wsBwdData,
        gpuPrm->wsBwdDataSz,
        &beta,
        gpuPrm->grout_desc,
        gpuPrm->d_grout));

    // результ
    cuCHECK(cudaMemcpy(gradOut, gpuPrm->d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
   
    if (gpuClearMem_){      
        cuCHECK(cudaFree(gpuPrm->d_w));          gpuPrm->d_w = 0;
        cuCHECK(cudaFree(d_grin));               gpuPrm->d_out = 0;
        cuCHECK(cudaFree(gpuPrm->d_grout));      gpuPrm->d_grout = 0;
        cuCHECK(cudaFree(gpuPrm->d_wsBwdData));  gpuPrm->d_wsBwdData = 0;
    }
}


#elif SN_CUDA

#include <cuda_runtime.h>
#include "../stdafx.h"
#include "SNOperator/src/Operator/convolution.h"

using namespace std;
using namespace SN_Base;

#ifndef cuCHECK
#define cuCHECK(func) if (func != 0){ ERROR_MESS("CUDA error: " + cudaGetErrorString(cudaGetLastError())); return;}
#endif


struct gpuParams{

    cudaDeviceProp* cu_deviceProps = 0;      
    
    snFloat* d_in = 0;
    snFloat* d_out = 0;
    snFloat* d_w = 0;
    snFloat* d_dw = 0;
    snFloat* d_grout = 0;   
};

void Convolution::iniParamCUDA(const snSize& insz, const snSize& outsz,
    const convParams& prms, void** pGpuPrm){

    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)*pGpuPrm;
    if (!gpuPrm){
        cudaDeviceProp* cu_deviceProps = new cudaDeviceProp();

        cudaGetDeviceProperties(cu_deviceProps, 0);
        if (cu_deviceProps->major < 2){
            ERROR_MESS("%s requires SM >= 2.0");
            delete cu_deviceProps;
            return;
        } 
        gpuPrm = new gpuParams();
        *pGpuPrm = gpuPrm;
        
        gpuPrm->cu_deviceProps = cu_deviceProps;
       
        if (!gpuClearMem_){
            cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm->d_w, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));  
            cuCHECK(cudaMalloc(&gpuPrm->d_grout, insz.size() * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm->d_dw, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat)));
        }
    }
    else if (!gpuClearMem_){

        cuCHECK(cudaFree(gpuPrm->d_in));    gpuPrm->d_in = 0;
        cuCHECK(cudaFree(gpuPrm->d_w));     gpuPrm->d_w = 0;
        cuCHECK(cudaFree(gpuPrm->d_out));   gpuPrm->d_out = 0;
        cuCHECK(cudaFree(gpuPrm->d_grout)); gpuPrm->d_grout = 0;
        cuCHECK(cudaFree(gpuPrm->d_dw));    gpuPrm->d_dw = 0;

        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_w, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_dw, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat)));
    }
}

void Convolution::freeParamCUDA(void* gpuPrms){
 
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;

    if (!gpuPrm) return;

    delete gpuPrm->cu_deviceProps;

    cuCHECK(cudaFree(gpuPrm->d_in));
    cuCHECK(cudaFree(gpuPrm->d_w));
    cuCHECK(cudaFree(gpuPrm->d_out));
    cuCHECK(cudaFree(gpuPrm->d_grout));
    cuCHECK(cudaFree(gpuPrm->d_dw));
}

__global__ void cuConvFwd(size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* output){
           
    size_t wStepByD = fWidth * fHeight,        // step weight by input
           wStepByK = wStepByD * insz.d + 1,   // step weight by output
           outStepByD = outsz.w * outsz.h,     // step out by input
           outStepByN = outStepByD * outsz.d,  // step out by batch
           inStepByD = insz.w * insz.h,        // step in by input
           inStepByN = inStepByD * insz.d;     // step in by batch

    // gridDim.x - number of out layers
    // gridDim.y - batch size
  
    weight += blockIdx.x * wStepByK;
    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;
    input += blockIdx.y * inStepByN;
       
 
    unsigned int oz = 0;
    while (oz < insz.d){
              
        unsigned int oy = threadIdx.y;
        while (oy < outsz.h){

            unsigned int ox = threadIdx.x;
            while (ox < outsz.w){

                if (oz == 0)
                    output[ox + oy * outsz.w] = weight[wStepByD * insz.d]; // bias

                size_t posW = ox * stride, posH = oy * stride;
                               
                // kernel
                snFloat csum = 0;
#pragma unroll
                for (size_t c = 0; c < wStepByD; ++c){

                    size_t cx = c % fWidth, cy = c / fWidth;

                    csum += input[(cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w] * weight[cx + cy * fWidth];
                }
                output[ox + oy * outsz.w] += csum;

                ox += blockDim.x;
            }
            oy += blockDim.y;
        }
        weight += wStepByD;
        input += inStepByD;
        ++oz;
    }
}

void Convolution::forwardCUDA(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, void* gpuPrms){
   
    cudaSetDevice(gpuDeviceId_);
  
    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));                              
        cuCHECK(cudaMalloc(&gpuPrm->d_w, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));                            
    }
    
    // input
    cuCHECK(cudaMemcpy(gpuPrm->d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
     
    // weight
    cuCHECK(cudaMemcpy(gpuPrm->d_w, weight, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));
             
    // run     
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(outsz.d), unsigned int(outsz.n));
  
    cuConvFwd <<< dimGrid, dimBlock >>>(prms.fWidth,
        prms.fHeight,
        prms.dilate,
        prms.stride,
        gpuPrm->d_w,
        insz, 
        gpuPrm->d_in,
        outsz, 
        gpuPrm->d_out);
    
    // result
    cuCHECK(cudaMemcpy(output, gpuPrm->d_out, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
    
    if (gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_in));   gpuPrm->d_in = 0;
        cuCHECK(cudaFree(gpuPrm->d_w));    gpuPrm->d_w = 0;
        cuCHECK(cudaFree(gpuPrm->d_out));  gpuPrm->d_out = 0;     
    }
}

__global__ void cuConvBwd_GW(size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){

    size_t wStepByD = fWidth * fHeight,        // step weight by input
           wStepByK = wStepByD * insz.d + 1,   // step weight by output
           wStepByN = wStepByK * outsz.d,      // step weight by batch
           outStepByD = outsz.w * outsz.h,     // step out by input 
           outStepByN = outStepByD * outsz.d,  // step out by batch
           inStepByD = insz.w * insz.h,        // step in by input
           inStepByN = inStepByD * insz.d;     // step in by batch

    // gridDim.x - number of input layers
    // gridDim.y - batch size
        
    input += blockIdx.x * inStepByD + blockIdx.y * inStepByN;
    weight += blockIdx.x * wStepByD;
    dWeightOut += blockIdx.x * wStepByD + blockIdx.y * wStepByN;
    gradIn += blockIdx.y * outStepByN;
    gradOut += blockIdx.x * inStepByD + blockIdx.y * inStepByN;

   
    unsigned int oz = 0;
    while (oz < outsz.d){
       
        memset(dWeightOut, 0, wStepByD * sizeof(snFloat));
        if (blockIdx.x == 0)
            dWeightOut[wStepByD * insz.d] = 0;

        unsigned int oy = threadIdx.y;
        while (oy < outsz.h){
                       
            unsigned int ox = threadIdx.x;
            while (ox < outsz.w){

                size_t posW = ox * stride, posH = oy * stride;
              
                if (oz == 0){
                    for (size_t c = 0; c < wStepByD; ++c){
                        size_t cx = c % fWidth, cy = c / fWidth;
                        gradOut[(cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w] = 0;
                    }
                }

                // kernel conv
                snFloat grIn = gradIn[ox + oy * outsz.w];
#pragma unroll
                for (size_t c = 0; c < wStepByD; ++c){

                    size_t cx = c % fWidth, cy = c / fWidth,
                        si = (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w,
                        sw = cx + cy * fWidth;

                    gradOut[si] += grIn * weight[sw];

                    dWeightOut[sw] += grIn * input[si];                    
                }        
                if (blockIdx.x == 0)
                    dWeightOut[wStepByD * insz.d] += grIn; // bias
                
                ox += blockDim.x;
            }
            oy += blockDim.y;
        }
        weight += wStepByK;
        dWeightOut += wStepByK;
        gradIn += outStepByD;
        ++oz;
    }
}

// weighted averaging over batch
__global__ void cuConvWeightMean(size_t kernel, size_t fWidth, size_t fHeight, snSize insz, snFloat* weight){

    size_t wStepByD = fWidth * fHeight,     // step weight by input
        wStepByK = wStepByD * insz.d + 1,   // step weight by output
        wStepByN = wStepByK * kernel;       // step weight by batch
        
    unsigned int ox = threadIdx.x;
    while (ox < wStepByN){

        snFloat csum = weight[ox];
        for (size_t i = 1; i < insz.n; ++i)
            csum += weight[ox + wStepByN * i];
               
        weight[ox] = csum / insz.n;

        ox += blockDim.x;
    }   
}

void Convolution::backwardCUDA_GW(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, void* gpuPrms){
  
    cudaSetDevice(gpuDeviceId_);
       
    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    snFloat* d_grin = gpuPrm->d_out;
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));                                         
        cuCHECK(cudaMalloc(&gpuPrm->d_w, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&d_grin, outsz.size() * sizeof(snFloat)));                                      
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, insz.size() * sizeof(snFloat)));                                      
        cuCHECK(cudaMalloc(&gpuPrm->d_dw, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat)));
    }

    // input
    cuCHECK(cudaMemcpy(gpuPrm->d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // weight
    cuCHECK(cudaMemcpy(gpuPrm->d_w, weight, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));
       
    // run   
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(insz.d), unsigned int(outsz.n));
   
    cuConvBwd_GW <<< dimGrid, dimBlock >>> (prms.fWidth,
        prms.fHeight, 
        prms.dilate,
        prms.stride,
        gpuPrm->d_w,
        insz, 
        gpuPrm->d_in,
        outsz, 
        d_grin,
        gpuPrm->d_grout,
        gpuPrm->d_dw);

    cuConvWeightMean <<< 1, 32 >>> (prms.kernel, prms.fWidth, prms.fHeight, insz, gpuPrm->d_dw);
   
    // result
    cuCHECK(cudaMemcpy(gradOut, gpuPrm->d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
    cuCHECK(cudaMemcpy(dWeightOut, gpuPrm->d_dw, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_in));      gpuPrm->d_in = 0;
        cuCHECK(cudaFree(gpuPrm->d_w));       gpuPrm->d_w = 0;
        cuCHECK(cudaFree(d_grin));            gpuPrm->d_out = 0;
        cuCHECK(cudaFree(gpuPrm->d_grout));   gpuPrm->d_grout = 0;
        cuCHECK(cudaFree(gpuPrm->d_dw));      gpuPrm->d_dw = 0;
    }
}

__global__ void cuConvBwd_G(size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, snSize insz, snSize outsz, snFloat* gradIn, snFloat* gradOut){

    size_t wStepByD = fWidth * fHeight,     // step weight by input
        wStepByK = wStepByD * insz.d + 1,   // step weight by output
        outStepByD = outsz.w * outsz.h,     // step out by input 
        outStepByN = outStepByD * outsz.d,  // step out by batch
        inStepByD = insz.w * insz.h,        // step in by input
        inStepByN = inStepByD * insz.d;     // step in by batch

    
    // gridDim.x - number of input layers
    // gridDim.y - batch size

    weight += blockIdx.x * wStepByD;
    gradIn += blockIdx.y * outStepByN;
    gradOut += blockIdx.x * inStepByD + blockIdx.y * inStepByN;
    
    unsigned int oz = 0;
    while (oz < outsz.d){
               
        unsigned int oy = threadIdx.y;
        while (oy < outsz.h){

            unsigned int ox = threadIdx.x;
            while (ox < outsz.w){

                size_t posW = ox * stride, posH = oy * stride;

                if (oz == 0){
                    for (size_t c = 0; c < wStepByD; ++c){
                        size_t cx = c % fWidth, cy = c / fWidth;
                        gradOut[(cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w] = 0;
                    }
                }

                // kernel conv
                snFloat grIn = gradIn[ox + oy * outsz.w];
#pragma unroll
                for (size_t c = 0; c < wStepByD; ++c){

                    size_t cx = c % fWidth, cy = c / fWidth,
                        si = (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w,
                        sw = cx + cy * fWidth;

                    gradOut[si] += grIn * weight[sw];
                }
                
                ox += blockDim.x;
            }
            oy += blockDim.y;
        }
        weight += wStepByK;
        gradIn += outStepByD;
        ++oz;
    }
}

void Convolution::backwardCUDA_G(const convParams& prms,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, void* gpuPrms){
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    snFloat* d_grin = gpuPrm->d_out;
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&d_grin, outsz.size() * sizeof(snFloat)));                          
        cuCHECK(cudaMalloc(&gpuPrm->d_w, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, insz.size() * sizeof(snFloat)));                          
    }

    // input
    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // weight
    cuCHECK(cudaMemcpy(gpuPrm->d_w, weight, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));
    
    // run
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(insz.d), unsigned int(outsz.n));

    cuConvBwd_G <<< dimGrid, dimBlock >>> (prms.fWidth,
        prms.fHeight, 
        prms.dilate,
        prms.stride,
        gpuPrm->d_w,
        insz, outsz, 
        d_grin, 
        gpuPrm->d_grout);
       
    // result
    cuCHECK(cudaMemcpy(gradOut, gpuPrm->d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_w));       gpuPrm->d_w = 0;
        cuCHECK(cudaFree(d_grin));            gpuPrm->d_out = 0;
        cuCHECK(cudaFree(gpuPrm->d_grout));   gpuPrm->d_grout = 0;
    }
}

#endif 
