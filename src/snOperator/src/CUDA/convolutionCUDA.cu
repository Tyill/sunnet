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


void Convolution::iniParamCUDA(const snSize& insz, const snSize& outsz, size_t fWidth, size_t fHeight, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    if (gpuPrm.find("cu_deviceProps") == gpuPrm.end()){

        cudaDeviceProp* cu_deviceProps = new cudaDeviceProp();

        cudaGetDeviceProperties(cu_deviceProps, 0);
        if (cu_deviceProps->major < 2){
            ERROR_MESS("%s requires SM >= 2.0");
            delete cu_deviceProps;
            return;
        }
        gpuPrm["cu_deviceProps"] = cu_deviceProps;

        if (!gpuClearMem_){
            snFloat *d_in = 0, *d_w = 0, *d_out = 0, *d_grout = 0, *d_dw = 0;
            cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));                                         gpuPrm["d_in"] = d_in;
            cuCHECK(cudaMalloc((void **)&d_w, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));            gpuPrm["d_w"] = d_w;
            cuCHECK(cudaMalloc((void **)&d_out, outsz.size() * sizeof(snFloat)));                                       gpuPrm["d_out"] = d_out;
            cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));                                      gpuPrm["d_grout"] = d_grout;
            cuCHECK(cudaMalloc((void **)&d_dw, (fWidth * fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat))); gpuPrm["d_dw"] = d_dw;
        }
    }
    else if (!gpuClearMem_){
        snFloat *d_in = (snFloat*)gpuPrm["d_in"],
            *d_w = (snFloat*)gpuPrm["d_w"],
            *d_out = (snFloat*)gpuPrm["d_out"],
            *d_grout = (snFloat*)gpuPrm["d_grout"],
            *d_dw = (snFloat*)gpuPrm["d_dw"];

        cuCHECK(cudaFree(d_in));    cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));                                         gpuPrm["d_in"] = d_in;
        cuCHECK(cudaFree(d_w));     cuCHECK(cudaMalloc((void **)&d_w, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));            gpuPrm["d_w"] = d_w;
        cuCHECK(cudaFree(d_out));   cuCHECK(cudaMalloc((void **)&d_out, outsz.size() * sizeof(snFloat)));                                       gpuPrm["d_out"] = d_out;
        cuCHECK(cudaFree(d_grout)); cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));                                      gpuPrm["d_grout"] = d_grout;
        cuCHECK(cudaFree(d_dw));    cuCHECK(cudaMalloc((void **)&d_dw, (fWidth * fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat))); gpuPrm["d_dw"] = d_dw;
    }
}

void Convolution::freeParamCUDA(map<std::string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    if (gpuPrm.find("cu_deviceProps") == gpuPrm.end()) return;

    delete (cudaDeviceProp*)gpuPrm["cu_deviceProps"];

    if (!gpuClearMem_){
        for (auto p : gpuPrm)
            if (p.first != "cu_deviceProps") cudaFree(p.second);
    }
}

void Convolution::forwardCUDA(size_t kernel, size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);


    cudnnHandle_t cudnn;
    cuCHECK(cudnnCreate(&cudnn));
    
    // input
    cudnnTensorDescriptor_t in_desc;
    cuCHECK(cudnnCreateTensorDescriptor(&in_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, insz.n, insz.d, insz.h, insz.w));

    snFloat *in_data;
    cuCHECK(cudaMalloc(&in_data, insz.size() * sizeof(snFloat)));

    // mask      
    cudnnFilterDescriptor_t filt_desc;
    cuCHECK(cudnnCreateFilterDescriptor(&filt_desc));
    cuCHECK(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        1, 1, fHeight, fWidth));

    snFloat *filt_data;
    cuCHECK(cudaMalloc(&filt_data, 1 * 1 * fHeight * fWidth * sizeof(snFloat)));

    // conv
    cudnnConvolutionDescriptor_t conv_desc;
    cuCHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    cuCHECK(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, stride, stride, dilate, dilate,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    // output
    int out_n, out_c, out_h, out_w;
    cuCHECK(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

    cudnnTensorDescriptor_t out_desc;
    cuCHECK(cudnnCreateTensorDescriptor(&out_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

    snFloat *out_data;
    cuCHECK(cudaMalloc(&out_data, out_n * out_c * out_h * out_w * sizeof(snFloat)));

    // algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cuCHECK(cudnnGetConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
        
    // workspace
    size_t ws_size;
    cuCHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

    snFloat *ws_data;
    cuCHECK(cudaMalloc(&ws_data, ws_size));
       
    // perform
    snFloat alpha = 1.f, beta = 0.f;
    //dev_iota << <in_w * in_h, in_n * in_c >> >(in_data);
    //dev_const << <filt_w * filt_h, filt_k * filt_c >> >(filt_data, 1.f);
    
    cuCHECK(cudnnConvolutionForward(cudnn, &alpha, in_desc, in_data, filt_desc, filt_data,
        conv_desc, algo, ws_data, ws_size,
        &beta, out_desc, out_data));

   
    // finalizing
    cuCHECK(cudaFree(ws_data));
    cuCHECK(cudaFree(out_data));
    cuCHECK(cudnnDestroyTensorDescriptor(out_desc));
    cuCHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    cuCHECK(cudaFree(filt_data));
    cuCHECK(cudnnDestroyFilterDescriptor(filt_desc));
    cuCHECK(cudaFree(in_data));
    cuCHECK(cudnnDestroyTensorDescriptor(in_desc));
    cuCHECK(cudnnDestroy(cudnn));   
}

__global__ void cuConvBwd_GW(size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){

    size_t wStepByD = fWidth * fHeight,        // шаг весов по входу
        wStepByK = wStepByD * insz.d + 1,   // шаг весов по выходу
        wStepByN = wStepByK * outsz.d,      // шаг весов по батчу
        outStepByD = outsz.w * outsz.h,     // шаг вых слоя по выходу
        outStepByN = outStepByD * outsz.d,  // шаг вых слоя по батчу
        inStepByD = insz.w * insz.h,        // шаг вх слоя по входу
        inStepByN = inStepByD * insz.d;     // шаг вх слоя по батчу

    // gridDim.x - кол-во вх слоев
    // gridDim.y - размер батча

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

                // ядро свертки   
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

// усреднение весов по батчу
__global__ void cuConvWeightMean(size_t kernel, size_t fWidth, size_t fHeight, snSize insz, snFloat* weight){

    size_t wStepByD = fWidth * fHeight,     // шаг весов по входу
        wStepByK = wStepByD * insz.d + 1,   // шаг весов по выходу
        wStepByN = wStepByK * kernel;       // шаг весов по батчу

    unsigned int ox = threadIdx.x;
    while (ox < wStepByN){

        snFloat csum = weight[ox];
        for (size_t i = 1; i < insz.n; ++i)
            csum += weight[ox + wStepByN * i];

        weight[ox] = csum / insz.n;

        ox += blockDim.x;
    }
}

void Convolution::backwardCUDA_GW(size_t kernel, size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    snFloat* d_in = (snFloat*)gpuPrm["d_in"],
        *d_grin = (snFloat*)gpuPrm["d_out"],
        *d_w = (snFloat*)gpuPrm["d_w"],
        *d_dw = (snFloat*)gpuPrm["d_dw"],
        *d_grout = (snFloat*)gpuPrm["d_grout"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_w, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_grin, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_dw, (fWidth * fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat)));
    }

    // вход данные
    cuCHECK(cudaMemcpy(d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // веса
    cuCHECK(cudaMemcpy(d_w, weight, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));


    // выполнение   
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(insz.d), unsigned int(outsz.n));

    cuConvBwd_GW << < dimGrid, dimBlock >> > (fWidth, fHeight, dilate, stride, d_w, insz, d_in, outsz, d_grin, d_grout, d_dw);

    cuConvWeightMean << < 1, 32 >> > (kernel, fWidth, fHeight, insz, d_dw);

    // результ
    cuCHECK(cudaMemcpy(gradOut, d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
    cuCHECK(cudaMemcpy(dWeightOut, d_dw, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(d_in));
        cuCHECK(cudaFree(d_w));
        cuCHECK(cudaFree(d_grin));
        cuCHECK(cudaFree(d_grout));
        cuCHECK(cudaFree(d_dw));
    }
}

__global__ void cuConvBwd_G(size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, snSize insz, snSize outsz, snFloat* gradIn, snFloat* gradOut){

    size_t wStepByD = fWidth * fHeight,        // шаг весов по входу
        wStepByK = wStepByD * insz.d + 1,   // шаг весов по выходу
        outStepByD = outsz.w * outsz.h,     // шаг вых слоя по выходу
        outStepByN = outStepByD * outsz.d,  // шаг вых слоя по батчу
        inStepByD = insz.w * insz.h,        // шаг вх слоя по входу
        inStepByN = inStepByD * insz.d;     // шаг вх слоя по батчу

    // gridDim.x - кол-во вх слоев
    // gridDim.y - размер батча

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

                // ядро свертки   
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

void Convolution::backwardCUDA_G(size_t kernel, size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    snFloat* d_grin = (snFloat*)gpuPrm["d_out"],
        *d_w = (snFloat*)gpuPrm["d_w"],
        *d_grout = (snFloat*)gpuPrm["d_grout"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc((void **)&d_grin, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_w, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));
    }

    // вход данные
    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // веса
    cuCHECK(cudaMemcpy(d_w, weight, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));

    // выход

    // выполнение   
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(insz.d), unsigned int(outsz.n));

    cuConvBwd_G << < dimGrid, dimBlock >> > (fWidth, fHeight, dilate, stride, d_w, insz, outsz, d_grin, d_grout);

    // результ
    cuCHECK(cudaMemcpy(gradOut, d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(d_w));
        cuCHECK(cudaFree(d_grin));
        cuCHECK(cudaFree(d_grout));
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

void Convolution::iniParamCUDA(const snSize& insz, const snSize& outsz, size_t fWidth, size_t fHeight, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    if (gpuPrm.find("cu_deviceProps") == gpuPrm.end()){

        cudaDeviceProp* cu_deviceProps = new cudaDeviceProp();

        cudaGetDeviceProperties(cu_deviceProps, 0);
        if (cu_deviceProps->major < 2){
            ERROR_MESS("%s requires SM >= 2.0");
            delete cu_deviceProps;
            return;
        }
        gpuPrm["cu_deviceProps"] = cu_deviceProps;

        if (!gpuClearMem_){
            snFloat *d_in = 0, *d_w = 0, *d_out = 0, *d_grout = 0, *d_dw = 0;
            cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));                                         gpuPrm["d_in"] = d_in;
            cuCHECK(cudaMalloc((void **)&d_w, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));            gpuPrm["d_w"] = d_w;
            cuCHECK(cudaMalloc((void **)&d_out, outsz.size() * sizeof(snFloat)));                                       gpuPrm["d_out"] = d_out;
            cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));                                      gpuPrm["d_grout"] = d_grout;
            cuCHECK(cudaMalloc((void **)&d_dw, (fWidth * fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat))); gpuPrm["d_dw"] = d_dw;
        }
    }
    else if (!gpuClearMem_){
        snFloat *d_in = (snFloat*)gpuPrm["d_in"],
            *d_w = (snFloat*)gpuPrm["d_w"],
            *d_out = (snFloat*)gpuPrm["d_out"],
            *d_grout = (snFloat*)gpuPrm["d_grout"],
            *d_dw = (snFloat*)gpuPrm["d_dw"];

        cuCHECK(cudaFree(d_in));    cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));                                         gpuPrm["d_in"] = d_in;
        cuCHECK(cudaFree(d_w));     cuCHECK(cudaMalloc((void **)&d_w, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));            gpuPrm["d_w"] = d_w;
        cuCHECK(cudaFree(d_out));   cuCHECK(cudaMalloc((void **)&d_out, outsz.size() * sizeof(snFloat)));                                       gpuPrm["d_out"] = d_out;
        cuCHECK(cudaFree(d_grout)); cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));                                      gpuPrm["d_grout"] = d_grout;
        cuCHECK(cudaFree(d_dw));    cuCHECK(cudaMalloc((void **)&d_dw, (fWidth * fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat))); gpuPrm["d_dw"] = d_dw;
    }
}

void Convolution::freeParamCUDA(map<std::string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    if (gpuPrm.find("cu_deviceProps") == gpuPrm.end()) return;

    delete (cudaDeviceProp*)gpuPrm["cu_deviceProps"];

    if (!gpuClearMem_){
        for (auto p : gpuPrm)
            if (p.first != "cu_deviceProps") cudaFree(p.second);
    }
}

__global__ void cuConvFwd(size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* output){
           
    size_t wStepByD = fWidth * fHeight,        // шаг весов по входу
           wStepByK = wStepByD * insz.d + 1,   // шаг весов по выходу
           outStepByD = outsz.w * outsz.h,     // шаг вых слоя по выходу
           outStepByN = outStepByD * outsz.d,  // шаг вых слоя по батчу
           inStepByD = insz.w * insz.h,        // шаг вх слоя по входу
           inStepByN = inStepByD * insz.d;     // шаг вх слоя по батчу

    // gridDim.x - кол-во вых слоев
    // gridDim.y - размер батча
  
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
                               
                // ядро свертки   
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

void Convolution::forwardCUDA(size_t kernel, size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    snFloat* d_in = (snFloat*)gpuPrm["d_in"],
           * d_w = (snFloat*)gpuPrm["d_w"],
           * d_out = (snFloat*)gpuPrm["d_out"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));                              
        cuCHECK(cudaMalloc((void **)&d_w, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat))); 
        cuCHECK(cudaMalloc((void **)&d_out, outsz.size() * sizeof(snFloat)));                            
    }
    
    // вход
    cuCHECK(cudaMemcpy(d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
     
    // веса
    cuCHECK(cudaMemcpy(d_w, weight, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));

     
    // выполнение     
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(outsz.d), unsigned int(outsz.n));
  
    cuConvFwd <<< dimGrid, dimBlock >>>(fWidth, fHeight, dilate, stride, d_w, insz, d_in, outsz, d_out);
    
    // результ
    cuCHECK(cudaMemcpy(output, d_out, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));  

    if (gpuClearMem_){
        cuCHECK(cudaFree(d_in));
        cuCHECK(cudaFree(d_w));
        cuCHECK(cudaFree(d_out));        
    }
}

__global__ void cuConvBwd_GW(size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){

    size_t wStepByD = fWidth * fHeight,        // шаг весов по входу
           wStepByK = wStepByD * insz.d + 1,   // шаг весов по выходу
           wStepByN = wStepByK * outsz.d,      // шаг весов по батчу
           outStepByD = outsz.w * outsz.h,     // шаг вых слоя по выходу
           outStepByN = outStepByD * outsz.d,  // шаг вых слоя по батчу
           inStepByD = insz.w * insz.h,        // шаг вх слоя по входу
           inStepByN = inStepByD * insz.d;     // шаг вх слоя по батчу

    // gridDim.x - кол-во вх слоев
    // gridDim.y - размер батча
        
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

                // ядро свертки   
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

// усреднение весов по батчу
__global__ void cuConvWeightMean(size_t kernel, size_t fWidth, size_t fHeight, snSize insz, snFloat* weight){

    size_t wStepByD = fWidth * fHeight,     // шаг весов по входу
        wStepByK = wStepByD * insz.d + 1,   // шаг весов по выходу
        wStepByN = wStepByK * kernel;       // шаг весов по батчу
        
    unsigned int ox = threadIdx.x;
    while (ox < wStepByN){

        snFloat csum = weight[ox];
        for (size_t i = 1; i < insz.n; ++i)
            csum += weight[ox + wStepByN * i];
               
        weight[ox] = csum / insz.n;

        ox += blockDim.x;
    }   
}

void Convolution::backwardCUDA_GW(size_t kernel, size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    snFloat* d_in = (snFloat*)gpuPrm["d_in"],
           * d_grin = (snFloat*)gpuPrm["d_out"],
           * d_w = (snFloat*)gpuPrm["d_w"],
           * d_dw = (snFloat*)gpuPrm["d_dw"],
           * d_grout = (snFloat*)gpuPrm["d_grout"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));                                         
        cuCHECK(cudaMalloc((void **)&d_w, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));            
        cuCHECK(cudaMalloc((void **)&d_grin, outsz.size() * sizeof(snFloat)));                                      
        cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));                                      
        cuCHECK(cudaMalloc((void **)&d_dw, (fWidth * fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat))); 
    }

    // вход данные
    cuCHECK(cudaMemcpy(d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // веса
    cuCHECK(cudaMemcpy(d_w, weight, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));

   
    // выполнение   
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(insz.d), unsigned int(outsz.n));
   
    cuConvBwd_GW <<< dimGrid, dimBlock >>> (fWidth, fHeight, dilate, stride, d_w, insz, d_in, outsz, d_grin, d_grout, d_dw);

    cuConvWeightMean <<< 1, 32 >>> (kernel, fWidth, fHeight, insz, d_dw);
   
    // результ
    cuCHECK(cudaMemcpy(gradOut, d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
    cuCHECK(cudaMemcpy(dWeightOut, d_dw, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(d_in));
        cuCHECK(cudaFree(d_w));
        cuCHECK(cudaFree(d_grin));
        cuCHECK(cudaFree(d_grout));
        cuCHECK(cudaFree(d_dw));
    }
}

__global__ void cuConvBwd_G(size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, snSize insz, snSize outsz, snFloat* gradIn, snFloat* gradOut){

    size_t wStepByD = fWidth * fHeight,        // шаг весов по входу
        wStepByK = wStepByD * insz.d + 1,   // шаг весов по выходу
        outStepByD = outsz.w * outsz.h,     // шаг вых слоя по выходу
        outStepByN = outStepByD * outsz.d,  // шаг вых слоя по батчу
        inStepByD = insz.w * insz.h,        // шаг вх слоя по входу
        inStepByN = inStepByD * insz.d;     // шаг вх слоя по батчу

    // gridDim.x - кол-во вх слоев
    // gridDim.y - размер батча

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

                // ядро свертки   
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

void Convolution::backwardCUDA_G(size_t kernel, size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    snFloat* d_grin = (snFloat*)gpuPrm["d_out"],
           * d_w = (snFloat*)gpuPrm["d_w"],
           * d_grout = (snFloat*)gpuPrm["d_grout"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc((void **)&d_grin, outsz.size() * sizeof(snFloat)));                          
        cuCHECK(cudaMalloc((void **)&d_w, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));                          
    }

    // вход данные
    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // веса
    cuCHECK(cudaMemcpy(d_w, weight, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));
        
    // выход

    // выполнение   
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(insz.d), unsigned int(outsz.n));

    cuConvBwd_G <<< dimGrid, dimBlock >>> (fWidth, fHeight, dilate, stride, d_w, insz, outsz, d_grin, d_grout);
       
    // результ
    cuCHECK(cudaMemcpy(gradOut, d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(d_w));
        cuCHECK(cudaFree(d_grin));
        cuCHECK(cudaFree(d_grout));
    }
}

#endif 
