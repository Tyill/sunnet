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

#ifdef SN_sCUDNN

#include <cuda_runtime.h>
#include <cudnn.h>
#include "../stdafx.h"
#include "SNOperator/src/Operator/convolution.h"

using namespace std;
using namespace SN_Base;

#ifndef cuCHECK
#define cuCHECK(func) if (func != 0){ ERROR_MESS("CUDA error: " + cudaGetErrorString(cudaGetLastError())); return;}
#endif


void Convolution::iniParamCUDA(const snSize& insz, const snSize& outsz, 
    size_t fWidth, size_t fHeight, size_t dilate, size_t stride, map<string, void*>& gpuPrm){

    cudaSetDevice(gpuDeviceId_);
    
    bool isFirst = false;

    cudnnHandle_t cudnn = nullptr;
   
    if (gpuPrm.find("cu_deviceProps") == gpuPrm.end()){

        cudaDeviceProp* cu_deviceProps = new cudaDeviceProp();

        cudaGetDeviceProperties(cu_deviceProps, 0);
        if (cu_deviceProps->major < 2){
            ERROR_MESS("%s requires SM >= 2.0");
            delete cu_deviceProps;
            return;
        }
        gpuPrm["cu_deviceProps"] = cu_deviceProps;
        
        cuCHECK(cudnnCreate(&cudnn));
        gpuPrm["cudnnHandle"] = cudnn;     

        gpuPrm["d_in"] = 0;  
        gpuPrm["d_w"] = 0;   
        gpuPrm["d_out"] = 0; 
        gpuPrm["d_ws"] = 0;  

        isFirst = true;
    }  
 
    // input
    cudnnTensorDescriptor_t in_desc = nullptr;
    cuCHECK(cudnnCreateTensorDescriptor(&in_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, insz.n, insz.d, insz.h, insz.w));
    if (!isFirst) 
        cuCHECK(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm["in_desc"]));
    gpuPrm["in_desc"] = in_desc;
    
    // mask      
    cudnnFilterDescriptor_t filt_desc = nullptr;
    cuCHECK(cudnnCreateFilterDescriptor(&filt_desc));
    cuCHECK(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        outsz.d, insz.d, fHeight, fWidth));
    if (!isFirst)
        cuCHECK(cudnnDestroyFilterDescriptor((cudnnFilterDescriptor_t)gpuPrm["filt_desc"]));
    gpuPrm["filt_desc"] = filt_desc;

    // conv
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cuCHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    cuCHECK(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, stride, stride, dilate, dilate,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    if (!isFirst)
        cuCHECK(cudnnDestroyConvolutionDescriptor((cudnnConvolutionDescriptor_t)gpuPrm["conv_desc"]));
    gpuPrm["conv_desc"] = conv_desc;

    // output
    int out_n = 0, out_c = 0, out_h = 0, out_w = 0;
    cuCHECK(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

    cudnnTensorDescriptor_t out_desc;
    cuCHECK(cudnnCreateTensorDescriptor(&out_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));
    if (!isFirst)
        cuCHECK(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm["out_desc"]));
    gpuPrm["out_desc"] = out_desc;

    // algorithm
    cudnnConvolutionFwdAlgo_t* algo = new cudnnConvolutionFwdAlgo_t();
    cuCHECK(cudnnGetConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algo));
    if (!isFirst)
        delete gpuPrm["algo"];
    gpuPrm["algo"] = algo;

    // workspace
    size_t* wsSz = new size_t();
    cuCHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, *algo, wsSz));
    if (!isFirst)
        delete gpuPrm["wsSz"];
    gpuPrm["wsSz"] = wsSz;

    if (isFirst && !gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm["d_in"], insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_w"], outsz.d * insz.d * fHeight * fWidth * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_out"], out_n * out_c * out_h * out_w * sizeof(snFloat)));  
        cuCHECK(cudaMalloc(&gpuPrm["d_ws"], *wsSz));
    }   
    else if (!gpuClearMem_){        
        cuCHECK(cudaFree(gpuPrm["d_in"]));  gpuPrm["d_in"] = 0; 
        cuCHECK(cudaFree(gpuPrm["d_w"]));   gpuPrm["d_w"] = 0;  
        cuCHECK(cudaFree(gpuPrm["d_out"])); gpuPrm["d_out"] = 0;
        cuCHECK(cudaFree(gpuPrm["d_ws"]));  gpuPrm["d_ws"] = 0;

        cuCHECK(cudaMalloc(&gpuPrm["d_in"], insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_w"], outsz.d * insz.d * fHeight * fWidth * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_out"], out_n * out_c * out_h * out_w * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_ws"], *wsSz));
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
    
    // вход
    cuCHECK(cudaMemcpy(gpuPrm["d_in"], input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // веса
    cuCHECK(cudaMemcpy(gpuPrm["d_w"], weight, outsz.d * insz.d * fHeight * fWidth * sizeof(snFloat), cudaMemcpyHostToDevice));
       
    snFloat alpha = 1.f, beta = 0.f;   
    cuCHECK(cudnnConvolutionForward((cudnnHandle_t)gpuPrm["cudnnHandle"],
        &alpha,
        (cudnnTensorDescriptor_t)gpuPrm["in_desc"],
        (snFloat*)gpuPrm["d_in"],
        (cudnnFilterDescriptor_t)gpuPrm["filt_desc"],
        (snFloat*)gpuPrm["d_w"],
        (cudnnConvolutionDescriptor_t)gpuPrm["conv_desc"],
        *(cudnnConvolutionFwdAlgo_t*)gpuPrm["algo"],
        (snFloat*)gpuPrm["d_ws"],
        *(size_t*)gpuPrm["wsSz"],
        &beta,
        (cudnnTensorDescriptor_t)gpuPrm["out_desc"],
        (snFloat*)gpuPrm["d_out"]));
 
    // результ
    cuCHECK(cudaMemcpy(output, (snFloat*)gpuPrm["d_out"], outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
      
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

void Convolution::iniParamCUDA(const snSize& insz, const snSize& outsz,
    const convParams& prms, map<string, void*>& gpuPrm){

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

        gpuPrm["d_in"] = 0;
        gpuPrm["d_w"] = 0;
        gpuPrm["d_out"] = 0;
        gpuPrm["d_grout"] = 0;
        gpuPrm["d_dw"] = 0;

        if (!gpuClearMem_){
            cuCHECK(cudaMalloc(&gpuPrm["d_in"], insz.size() * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm["d_w"], (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm["d_out"], outsz.size() * sizeof(snFloat)));  
            cuCHECK(cudaMalloc(&gpuPrm["d_grout"], insz.size() * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm["d_dw"], (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat)));
        }
    }
    else if (!gpuClearMem_){

        cuCHECK(cudaFree(gpuPrm["d_in"]));    gpuPrm["d_in"] = 0;
        cuCHECK(cudaFree(gpuPrm["d_w"]));     gpuPrm["d_w"] = 0;
        cuCHECK(cudaFree(gpuPrm["d_out"]));   gpuPrm["d_out"] = 0;
        cuCHECK(cudaFree(gpuPrm["d_grout"])); gpuPrm["d_grout"] = 0;
        cuCHECK(cudaFree(gpuPrm["d_dw"]));    gpuPrm["d_dw"] = 0;

        cuCHECK(cudaMalloc(&gpuPrm["d_in"], insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_w"], (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_out"], outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_grout"], insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_dw"], (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat)));
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

void Convolution::forwardCUDA(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, map<string, void*>& gpuPrm){
   
    cudaSetDevice(gpuDeviceId_);

    snFloat* d_in = (snFloat*)gpuPrm["d_in"],
           * d_w = (snFloat*)gpuPrm["d_w"],
           * d_out = (snFloat*)gpuPrm["d_out"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));                              
        cuCHECK(cudaMalloc((void **)&d_w, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_out, outsz.size() * sizeof(snFloat)));                            
    }
    
    // input
    cuCHECK(cudaMemcpy(d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
     
    // weight
    cuCHECK(cudaMemcpy(d_w, weight, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));
             
    // run     
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(outsz.d), unsigned int(outsz.n));
  
    cuConvFwd <<< dimGrid, dimBlock >>>(prms.fWidth, prms.fHeight, prms.dilate, prms.stride,
        d_w, insz, d_in, outsz, d_out);
    
    // result
    cuCHECK(cudaMemcpy(output, d_out, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));  

    if (gpuClearMem_){
        cuCHECK(cudaFree(d_in));
        cuCHECK(cudaFree(d_w));
        cuCHECK(cudaFree(d_out));        
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
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, map<string, void*>& gpuPrm){
  
    cudaSetDevice(gpuDeviceId_);

    snFloat* d_in = (snFloat*)gpuPrm["d_in"],
           * d_grin = (snFloat*)gpuPrm["d_out"],
           * d_w = (snFloat*)gpuPrm["d_w"],
           * d_dw = (snFloat*)gpuPrm["d_dw"],
           * d_grout = (snFloat*)gpuPrm["d_grout"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));                                         
        cuCHECK(cudaMalloc((void **)&d_w, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_grin, outsz.size() * sizeof(snFloat)));                                      
        cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));                                      
        cuCHECK(cudaMalloc((void **)&d_dw, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat)));
    }

    // input
    cuCHECK(cudaMemcpy(d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // weight
    cuCHECK(cudaMemcpy(d_w, weight, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));
       
    // run   
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(insz.d), unsigned int(outsz.n));
   
    cuConvBwd_GW <<< dimGrid, dimBlock >>> (prms.fWidth, prms.fHeight, prms.dilate, prms.stride,
        d_w, insz, d_in, outsz, d_grin, d_grout, d_dw);

    cuConvWeightMean <<< 1, 32 >>> (prms.kernel, prms.fWidth, prms.fHeight, insz, d_dw);
   
    // result
    cuCHECK(cudaMemcpy(gradOut, d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
    cuCHECK(cudaMemcpy(dWeightOut, d_dw, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyDeviceToHost));

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
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    snFloat* d_grin = (snFloat*)gpuPrm["d_out"],
           * d_w = (snFloat*)gpuPrm["d_w"],
           * d_grout = (snFloat*)gpuPrm["d_grout"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc((void **)&d_grin, outsz.size() * sizeof(snFloat)));                          
        cuCHECK(cudaMalloc((void **)&d_w, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));                          
    }

    // input
    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // weight
    cuCHECK(cudaMemcpy(d_w, weight, (prms.fWidth * prms.fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));
    
    // run
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(insz.d), unsigned int(outsz.n));

    cuConvBwd_G <<< dimGrid, dimBlock >>> (prms.fWidth, prms.fHeight, prms.dilate, prms.stride,
        d_w, insz, outsz, d_grin, d_grout);
       
    // result
    cuCHECK(cudaMemcpy(gradOut, d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(d_w));
        cuCHECK(cudaFree(d_grin));
        cuCHECK(cudaFree(d_grout));
    }
}

#endif 
