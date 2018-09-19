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

#include <cuda_runtime.h>
#include "../stdafx.h"
#include "SNOperator/src/Operator/deconvolution.h"

using namespace std;
using namespace SN_Base;

#ifndef cuCHECK
#define cuCHECK(func) if (func != 0){ ERROR_MESS("CUDA error: " + cudaGetErrorString(cudaGetLastError())); return;}
#endif

void Deconvolution::iniParamCUDA(snSize insz, snSize outsz, size_t fWidth, size_t fHeight, map<string, void*>& gpuPrm){

    if (gpuPrm.find("cu_deviceProps") == gpuPrm.end()){
       
        cudaDeviceProp* cu_deviceProps = new cudaDeviceProp();

        cudaGetDeviceProperties(cu_deviceProps, 0);
        if (cu_deviceProps->major < 2){
            ERROR_MESS("%s requires SM >= 2.0");
            delete cu_deviceProps;
            return;
        }
        gpuPrm["cu_deviceProps"] = cu_deviceProps;

        snFloat *d_in = 0, *d_w = 0, *d_out = 0, *d_grout = 0, *d_dw = 0;
        cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));                                         gpuPrm["d_in"] = d_in; 
        cuCHECK(cudaMalloc((void **)&d_w, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));            gpuPrm["d_w"] = d_w;
        cuCHECK(cudaMalloc((void **)&d_out, outsz.size() * sizeof(snFloat)));                                       gpuPrm["d_out"] = d_out;
        cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));                                      gpuPrm["d_grout"] = d_grout;
        cuCHECK(cudaMalloc((void **)&d_dw, (fWidth * fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat))); gpuPrm["d_dw"] = d_dw;        
    }
    else{
        snFloat *d_in =    (snFloat*)gpuPrm["d_in"],
                *d_w =     (snFloat*)gpuPrm["d_w"],
                *d_out =   (snFloat*)gpuPrm["d_out"],
                *d_grout = (snFloat*)gpuPrm["d_grout"],
                *d_dw =    (snFloat*)gpuPrm["d_dw"];

        cuCHECK(cudaFree(d_in));    cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));                                         gpuPrm["d_in"] = d_in;
        cuCHECK(cudaFree(d_w));     cuCHECK(cudaMalloc((void **)&d_w, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat)));            gpuPrm["d_w"] = d_w;
        cuCHECK(cudaFree(d_out));   cuCHECK(cudaMalloc((void **)&d_out, outsz.size() * sizeof(snFloat)));                                       gpuPrm["d_out"] = d_out;
        cuCHECK(cudaFree(d_grout)); cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));                                      gpuPrm["d_grout"] = d_grout;
        cuCHECK(cudaFree(d_dw));    cuCHECK(cudaMalloc((void **)&d_dw, (fWidth * fHeight * insz.d + 1) * outsz.d * outsz.n * sizeof(snFloat))); gpuPrm["d_dw"] = d_dw;
    }
}

void Deconvolution::freeParamCUDA(map<std::string, void*>& gpuPrm){

    if (gpuPrm.find("cu_deviceProps") == gpuPrm.end()) return;

    delete (cudaDeviceProp*)gpuPrm["cu_deviceProps"];

    for (auto p : gpuPrm)
        if (p.first != "cu_deviceProps") cudaFree(p.second);
}

__global__ void cuDeconvFwd(size_t fWidth, size_t fHeight, size_t stride,
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

                    csum += input[(cx + posW) + (cy + posH) * insz.w] * weight[cx + cy * fWidth];
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

void Deconvolution::forwardCUDA(size_t kernel, size_t fWidth, size_t fHeight, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* output, map<string, void*>& gpuPrm){

    // вход данные
    snFloat* d_in = (snFloat*)gpuPrm["d_in"];
    cuCHECK(cudaMemcpy(d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
     
    // веса
    snFloat* d_w = (snFloat*)gpuPrm["d_w"];
    cuCHECK(cudaMemcpy(d_w, weight, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));

    // выход
    snFloat* d_out = (snFloat*)gpuPrm["d_out"];
 
    // выполнение     
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(outsz.d), unsigned int(outsz.n));
  
    cuDeconvFwd <<< dimGrid, dimBlock >>>(fWidth, fHeight, stride, d_w, insz, d_in, outsz, d_out);
    
    // результ
    cuCHECK(cudaMemcpy(output, d_out, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));  
}

__global__ void cuDeconvBwd_GW(size_t fWidth, size_t fHeight, size_t stride,
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
                        gradOut[(cx + posW) + (cy + posH) * insz.w] = 0;
                    }
                }

                // ядро свертки   
                snFloat grIn = gradIn[ox + oy * outsz.w];
#pragma unroll
                for (size_t c = 0; c < wStepByD; ++c){

                    size_t cx = c % fWidth, cy = c / fWidth,
                        si = (cx + posW) + (cy + posH) * insz.w,
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
__global__ void cuDeconvWeightMean(size_t kernel, size_t fWidth, size_t fHeight, snSize insz, snFloat* weight){

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

void Deconvolution::backwardCUDA_GW(size_t kernel, size_t fWidth, size_t fHeight, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, map<string, void*>& gpuPrm){
       
    // вход данные
    snFloat* d_in = (snFloat*)gpuPrm["d_in"];
    cuCHECK(cudaMemcpy(d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    snFloat* d_grin = (snFloat*)gpuPrm["d_out"];
    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // веса
    snFloat* d_w = (snFloat*)gpuPrm["d_w"];
    cuCHECK(cudaMemcpy(d_w, weight, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));

    snFloat* d_dw = (snFloat*)gpuPrm["d_dw"];

    // выход
    snFloat* d_grout = (snFloat*)gpuPrm["d_grout"];

    // выполнение   
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(insz.d), unsigned int(outsz.n));
   
    cuDeconvBwd_GW <<< dimGrid, dimBlock >>> (fWidth, fHeight, stride, d_w, insz, d_in, outsz, d_grin, d_grout, d_dw);

    cuDeconvWeightMean <<< 1, 32 >>> (kernel, fWidth, fHeight, insz, d_dw);
   
    // результ
    cuCHECK(cudaMemcpy(gradOut, d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
    cuCHECK(cudaMemcpy(dWeightOut, d_dw, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyDeviceToHost));

}

__global__ void cuDeconvBwd_G(size_t fWidth, size_t fHeight, size_t stride,
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
                        gradOut[(cx + posW) + (cy + posH) * insz.w] = 0;
                    }
                }

                // ядро свертки   
                snFloat grIn = gradIn[ox + oy * outsz.w];
                for (size_t c = 0; c < wStepByD; ++c){

                    size_t cx = c % fWidth, cy = c / fWidth,
                        si = (cx + posW) + (cy + posH) * insz.w,
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

void Deconvolution::backwardCUDA_G(size_t kernel, size_t fWidth, size_t fHeight, size_t stride,
    snFloat* weight, snSize insz, snSize outsz, snFloat* gradIn, snFloat* gradOut, map<string, void*>& gpuPrm){

    // вход данные
    snFloat* d_grin = (snFloat*)gpuPrm["d_out"];
    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // веса
    snFloat* d_w = (snFloat*)gpuPrm["d_w"];
    cuCHECK(cudaMemcpy(d_w, weight, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice));
        
    // выход
    snFloat* d_grout = (snFloat*)gpuPrm["d_grout"];

    // выполнение   
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(insz.d), unsigned int(outsz.n));

    cuDeconvBwd_G <<< dimGrid, dimBlock >>> (fWidth, fHeight, stride, d_w, insz, outsz, d_grin, d_grout);
       
    // результ
    cuCHECK(cudaMemcpy(gradOut, d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
}

#endif 
