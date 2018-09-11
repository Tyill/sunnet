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

void Pooling::iniParamCUDA(snSize insz, snSize outsz, size_t kernel, map<string, void*>& gpuPrm){

    if (gpuPrm.find("cu_deviceProps") == gpuPrm.end()){

        cudaDeviceProp* cu_deviceProps = new cudaDeviceProp();
              
        cudaGetDeviceProperties(cu_deviceProps, 0);
        if (cu_deviceProps->major < 2){
            ERROR_MESS("%s requires SM >= 2.0");
            delete cu_deviceProps;
            return;
        }
        gpuPrm["cu_deviceProps"] = cu_deviceProps;

        snFloat* d_in = 0,* d_out = 0; size_t* d_idx = 0;
        cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));   gpuPrm["d_in"] = d_in;
        cuCHECK(cudaMalloc((void **)&d_out, outsz.size() * sizeof(snFloat))); gpuPrm["d_out"] = d_out;
        cuCHECK(cudaMalloc((void **)&d_idx, outsz.size() * sizeof(size_t)));  gpuPrm["d_idx"] = d_idx;
    }
    else{
        snFloat* d_in = (snFloat*)gpuPrm["d_in"],
               * d_out = (snFloat*)gpuPrm["d_out"],
               * d_idx = (snFloat*)gpuPrm["d_idx"];
            
        cuCHECK(cudaFree(d_in));  cuCHECK(cudaMalloc((void **)&d_in, insz.size() * sizeof(snFloat)));   gpuPrm["d_in"] = d_in;
        cuCHECK(cudaFree(d_out)); cuCHECK(cudaMalloc((void **)&d_out, outsz.size() * sizeof(snFloat))); gpuPrm["d_out"] = d_out;
        cuCHECK(cudaFree(d_idx)); cuCHECK(cudaMalloc((void **)&d_idx, outsz.size() * sizeof(size_t)));  gpuPrm["d_idx"] = d_idx;
    }
}

void Pooling::freeParamCUDA(map<std::string, void*>& gpuPrm){

    if (gpuPrm.find("cu_deviceProps") == gpuPrm.end()) return;

    delete (cudaDeviceProp*)gpuPrm["cu_deviceProps"];

    for (auto p : gpuPrm)
        if (p.first != "cu_deviceProps") cudaFree(p.second);
}


__global__ void cuPoolFwd(poolType type, size_t kernel, snSize insz, snFloat* input, snSize outsz, snFloat* output, size_t* outputInx){

    size_t outStepByD = outsz.w * outsz.h, // шаг вых слоя по выходу
        outStepByN = outStepByD * outsz.d, // шаг вых слоя по батчу
        inStepByD = insz.w * insz.h,       // шаг вх слоя по входу
        inStepByN = inStepByD * insz.d,    // шаг вх слоя по батчу
        kernelSz = kernel * kernel;     

    // gridDim.x - кол-во вх слоев
    // gridDim.y - кол-во вых слоев
    // gridDim.z - размер батча

    input += blockIdx.x * inStepByD + blockIdx.z * inStepByN;
    output += blockIdx.y * outStepByD + blockIdx.z * outStepByN;
    
    if (type == poolType::max){ // max

        unsigned int oz = threadIdx.z;
        while (oz < outsz.d){

            unsigned int oy = threadIdx.y;
            while (oy < outsz.h){

                unsigned int ox = threadIdx.x;
                while (ox < outsz.w){
                                       
                    size_t posW = ox * kernel, posH = oy * kernel;

                    // ядро свертки   
                    snFloat valMax = 0, idx = 0;
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
            input += inStepByD;
            output += outStepByD;
            outputInx += outStepByD;
            ++oz;
        }
    }
    else{ // mean

        unsigned int oz = 0;
        while (oz < insz.d){

            unsigned int oy = threadIdx.y;
            while (oy < outsz.h){

                unsigned int ox = threadIdx.x;
                while (ox < outsz.w){
                                        
                    size_t posW = ox * kernel, posH = oy * kernel;

                    // ядро свертки   
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
            input += inStepByD;
            output += outStepByD;
            ++oz;
        }
    }
}

void Pooling::forwardCUDA(poolType type, size_t kernel, snSize insz, snFloat* input,
    snSize outsz, snFloat* output, size_t* outputInx, map<string, void*>& gpuPrm){

    // вход данные
    snFloat* d_in = (snFloat*)gpuPrm["d_in"];
    cuCHECK(cudaMemcpy(d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
        
    // выход
    snFloat* d_out = (snFloat*)gpuPrm["d_out"];
    size_t* d_idx = (size_t*)gpuPrm["d_idx"];

    // выполнение     
    dim3 dimBlock(16, 16);
    dim3 dimGrid(outsz.d, outsz.n);

    cuPoolFwd <<< dimGrid, dimBlock >>>(type, kernel, insz, d_in, outsz, d_out, d_idx);

    // результ
    cuCHECK(cudaMemcpy(output, d_out, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
    cuCHECK(cudaMemcpy(outputInx, d_idx, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
}

__global__ void cuPoolBwd(poolType type, size_t kernel, snSize outsz, size_t* outputInx, snFloat* gradIn, snSize insz, snFloat* gradOut){

    size_t outStepByD = outsz.w * outsz.h,  // шаг вых слоя по выходу
        outStepByN = outStepByD * outsz.d,  // шаг вых слоя по батчу
        inStepByD = insz.w * insz.h,        // шаг вх слоя по входу
        inStepByN = inStepByD * insz.d;     // шаг вх слоя по батчу

    // gridDim.x - кол-во вх слоев
    // gridDim.y - размер батча

    gradIn += blockIdx.y * outStepByN;
    gradOut += blockIdx.x * inStepByD + blockIdx.y * inStepByN;

    if (type == poolType::max){ // max

        //unsigned int oz = 0;
        //while (oz < outsz.d){

        //    unsigned int oy = threadIdx.y;
        //    while (oy < outsz.h){

        //        unsigned int ox = threadIdx.x;
        //        while (ox < outsz.w){

        //            size_t posW = ox * stride, posH = oy * stride;

        //            if (oz == 0){
        //                for (size_t c = 0; c < wStepByD; ++c){
        //                    size_t cx = c % fWidth, cy = c / fWidth;
        //                    gradOut[(cx + posW) + (cy + posH) * insz.w] = 0;
        //                }
        //            }

        //            // ядро свертки   
        //            snFloat grIn = gradIn[ox + oy * outsz.w];
        //            for (size_t c = 0; c < wStepByD; ++c){

        //                size_t cx = c % fWidth, cy = c / fWidth,
        //                    si = cx + posW + (cy + posH) * insz.w, sw = cx + cy * fWidth;

        //                gradOut[si] += grIn * weight[sw];
        //            }

        //            ox += blockDim.x;
        //        }
        //        oy += blockDim.y;
        //    }
        //    gradIn += outStepByD;
        //    ++oz;
        //}
    }
    else{ // mean



    }
}

void Pooling::backwardCUDA(poolType type, size_t kernel, snSize outsz, size_t* outputInx, snFloat* gradIn,
    snSize insz, snFloat* gradOut, map<string, void*>& gpuPrm){

    // вход данные
    snFloat* d_grin = (snFloat*)gpuPrm["d_out"];
    cuCHECK(cudaMemcpy(d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    snFloat* d_idx = (snFloat*)gpuPrm["d_idx"];
    cuCHECK(cudaMemcpy(d_idx, outputInx, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
    
    // выход
    snFloat* d_grout = (snFloat*)gpuPrm["d_in"];
   
    // выполнение     
    dim3 dimBlock(16, 16);
    dim3 dimGrid(insz.d, outsz.n);

//    cuPoolBwd <<< dimGrid, dimBlock >>>(type, kernel, outsz, d_grin, insz, d_grout);

    // результ
    cuCHECK(cudaMemcpy(gradOut, d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
}

#endif 