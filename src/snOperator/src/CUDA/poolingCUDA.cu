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
#include "SNOperator/src/Operator/pooling.h"

using namespace std;
using namespace SN_Base;

#ifndef cuCHECK
#define cuCHECK(func) if (func != 0){ ERROR_MESS("CUDA error: " + cudaGetErrorString(cudaGetLastError())); return;}
#endif

void Pooling::iniParamCUDA(const snSize& insz, const snSize& outsz, size_t kernel, map<string, void*>& gpuPrm){
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
        gpuPrm["d_out"] = 0;
        gpuPrm["d_idx"] = 0;

        if (!gpuClearMem_){
            cuCHECK(cudaMalloc(&gpuPrm["d_in"], insz.size() * sizeof(snFloat))); 
            cuCHECK(cudaMalloc(&gpuPrm["d_out"], outsz.size() * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm["d_idx"], outsz.size() * sizeof(size_t)));
        }
    }
    else if (!gpuClearMem_){
                   
        cuCHECK(cudaFree(gpuPrm["d_in"]));  gpuPrm["d_in"] = 0;
        cuCHECK(cudaFree(gpuPrm["d_out"])); gpuPrm["d_out"] = 0;
        cuCHECK(cudaFree(gpuPrm["d_idx"])); gpuPrm["d_idx"] = 0;

        cuCHECK(cudaMalloc(&gpuPrm["d_in"], insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_out"], outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_idx"], outsz.size() * sizeof(size_t)));
    }
}

void Pooling::freeParamCUDA(map<std::string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    if (gpuPrm.find("cu_deviceProps") == gpuPrm.end()) return;

    delete (cudaDeviceProp*)gpuPrm["cu_deviceProps"];

    if (!gpuClearMem_){
        for (auto p : gpuPrm)
            if (p.first != "cu_deviceProps") cudaFree(p.second);
    }
}

__global__ void cuPoolFwd(poolType type, size_t kernel, snSize insz, snFloat* input, snSize outsz, snFloat* output, size_t* outputInx){

    size_t outStepByD = outsz.w * outsz.h, // step out by output
        outStepByN = outStepByD * outsz.d, // step out by batch
        inStepByD = insz.w * insz.h,       // step in by input
        inStepByN = inStepByD * insz.d,    // step in by batch
        kernelSz = kernel * kernel;     

    // gridDim.x - number of input layers
    // gridDim.y - batch sz
        
    input += blockIdx.x * inStepByD + blockIdx.y * inStepByN;
    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;
    outputInx += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    if (type == poolType::max){ // max
              
        unsigned int oy = threadIdx.y;
        while (oy < outsz.h){

            unsigned int ox = threadIdx.x;
            while (ox < outsz.w){
                                       
                size_t posW = ox * kernel, posH = oy * kernel;

                // kernel
                snFloat valMax = 0; size_t idx = 0;
#pragma unroll
                for (size_t c = 0; c < kernelSz; ++c){

                    size_t cx = c % kernel, cy = c / kernel;
                                                
                    snFloat val = input[cx + posW + (cy + posH) * insz.w];
                    if (val > valMax){
                        valMax = val;
                        idx = c;
                    }
                }
                output[ox + oy * outsz.w] = valMax;
                outputInx[ox + oy * outsz.w] = idx;

                ox += blockDim.x;
            }
            oy += blockDim.y;
        }           
    }
    else{ // mean
               
        unsigned int oy = threadIdx.y;
        while (oy < outsz.h){

            unsigned int ox = threadIdx.x;
            while (ox < outsz.w){
                                        
                size_t posW = ox * kernel, posH = oy * kernel;

                // kernel
                snFloat valMean = 0;
#pragma unroll
                for (size_t c = 0; c < kernelSz; ++c){

                    size_t cx = c % kernel, cy = c / kernel;

                    valMean += input[cx + posW + (cy + posH) * insz.w];
                }
                output[ox + oy * outsz.w] = valMean / kernelSz;
                   
                ox += blockDim.x;
            }
            oy += blockDim.y;
        }           
    }
}

void Pooling::forwardCUDA(poolType type, size_t kernel, const snSize& insz, snFloat* input,
    const snSize& outsz, snFloat* output, size_t* outputInx, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    snFloat* d_in = (snFloat*)gpuPrm["d_in"],
           * d_out = (snFloat*)gpuPrm["d_out"];
    size_t* d_idx = (size_t*)gpuPrm["d_idx"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));  
        cuCHECK(cudaMalloc((void **)&d_out, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_idx, outsz.size() * sizeof(size_t))); 
    }

    // input
    cuCHECK(cudaMemcpy(d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
    
    // run     
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(outsz.d), unsigned int(outsz.n));

    cuPoolFwd <<< dimGrid, dimBlock >>>(type, kernel, insz, d_in, outsz, d_out, d_idx);

    // result
    cuCHECK(cudaMemcpy(output, d_out, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
    cuCHECK(cudaMemcpy(outputInx, d_idx, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(d_in));
        cuCHECK(cudaFree(d_out));
        cuCHECK(cudaFree(d_idx));
    }
}

__global__ void cuPoolBwd(poolType type, size_t kernel, snSize outsz, size_t* outputInx, snFloat* gradIn, snSize insz, snFloat* gradOut){

    size_t outStepByD = outsz.w * outsz.h,     // step out by output
           outStepByN = outStepByD * outsz.d,  // step out by batch
           inStepByD = insz.w * insz.h,        // step in by input
           inStepByN = inStepByD * insz.d,     // step in by batch
           kernelSz = kernel * kernel;
    
    // gridDim.x - number of input layers
    // gridDim.y - batch sz

    gradIn += blockIdx.x * outStepByD + blockIdx.y * outStepByN;
    gradOut += blockIdx.x * inStepByD + blockIdx.y * inStepByN;
   
    if (type == poolType::max){ // max
        
        outputInx += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

        unsigned int oy = threadIdx.y;
        while (oy < outsz.h){

            unsigned int ox = threadIdx.x;
            while (ox < outsz.w){

                size_t posW = ox * kernel, posH = oy * kernel;
#pragma unroll
                for (size_t c = 0; c < kernelSz; ++c){
                    size_t cx = c % kernel, cy = c / kernel;
                    gradOut[(cx + posW) + (cy + posH) * insz.w] = 0;
                }
                                
                size_t c = outputInx[ox + oy * outsz.w], cx = c % kernel, cy = c / kernel;
                                      
                gradOut[cx + posW + (cy + posH) * insz.w] = gradIn[ox + oy * outsz.w];
                
                ox += blockDim.x;
            }
            oy += blockDim.y;
        }
    }
    else{ // mean

        unsigned int oy = threadIdx.y;
        while (oy < outsz.h){

            unsigned int ox = threadIdx.x;
            while (ox < outsz.w){

                size_t posW = ox * kernel, posH = oy * kernel;

                snFloat mean = gradIn[ox + oy * outsz.w] / kernel;
#pragma unroll
                for (size_t c = 0; c < kernelSz; ++c){
                    size_t cx = c % kernel, cy = c / kernel;
                    gradOut[(cx + posW) + (cy + posH) * insz.w] = mean;
                }
                                
                ox += blockDim.x;
            }
            oy += blockDim.y;
        }
    }
}

void Pooling::backwardCUDA(poolType type, size_t kernel, const snSize& outsz, size_t* outputInx, snFloat* gradIn,
    const snSize& insz, snFloat* gradOut, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    snFloat* d_grin = (snFloat*)gpuPrm["d_out"],
           * d_grout = (snFloat*)gpuPrm["d_in"];
    size_t* d_idx = (size_t*)gpuPrm["d_idx"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc((void **)&d_grin, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_grout, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc((void **)&d_idx, outsz.size() * sizeof(size_t)));
    }

    // input   
    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
      
    cuCHECK(cudaMemcpy(d_idx, outputInx, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
  
    // run     
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(outsz.d), unsigned int(outsz.n));

    cuPoolBwd <<< dimGrid, dimBlock >>>(type, kernel, outsz, d_idx, d_grin, insz, d_grout);

    // result
    cuCHECK(cudaMemcpy(gradOut, d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(d_grin));
        cuCHECK(cudaFree(d_grout));
        cuCHECK(cudaFree(d_idx));
    }
}

#endif 