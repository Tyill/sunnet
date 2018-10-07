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
#include "SNOperator/src/Operator/pooling.h"

using namespace std;
using namespace SN_Base;

#ifndef cuCHECK
#define cuCHECK(func) if (func != 0){ ERROR_MESS("CUDA error: " + cudaGetErrorString(cudaGetLastError())); return;}
#endif

struct gpuParams{

    cudnnHandle_t cudnn = 0;
    cudaDeviceProp* cu_deviceProps = 0;
    cudnnPoolingDescriptor_t pool_desc = 0;
    cudnnTensorDescriptor_t in_desc = 0;
    cudnnTensorDescriptor_t out_desc = 0;
    cudnnTensorDescriptor_t grin_desc = 0;
    cudnnTensorDescriptor_t grout_desc = 0;    
        
    snFloat* d_in = 0;   
    snFloat* d_out = 0;
    snFloat* d_grin = 0;
    snFloat* d_grout = 0;
};

void Pooling::iniParamCUDA(const snSize& insz, const snSize& outsz, size_t kernel, void** pGpuPrm){
    
    cudaSetDevice(gpuDeviceId_);
  
    bool isFirst = false;

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
       
        cudnnHandle_t cudnn = nullptr;
        cuCHECK(cudnnCreate(&cudnn));
        gpuPrm->cudnn = cudnn;              

        isFirst = true;
    }

    // input
    cudnnTensorDescriptor_t in_desc = nullptr;
    cuCHECK(cudnnCreateTensorDescriptor(&in_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, insz.n, insz.d, insz.h, insz.w));
    if (!isFirst)
        cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->in_desc));
    gpuPrm->in_desc = in_desc;

    // grout
    cudnnTensorDescriptor_t grout_desc;
    cuCHECK(cudnnCreateTensorDescriptor(&grout_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(grout_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, insz.n, insz.d, insz.h, insz.w));
    if (!isFirst)
        cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->grout_desc));
    gpuPrm->grout_desc = grout_desc;
      
    // pool
    cudnnPoolingDescriptor_t pool_desc = nullptr;
    cuCHECK(cudnnCreatePoolingDescriptor(&pool_desc));

    cudnnPoolingMode_t poolType = cudnnPoolingMode_t::CUDNN_POOLING_MAX;
    if (poolType_ == poolType::avg) 
        poolType = cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
   
    cuCHECK(cudnnSetPooling2dDescriptor(pool_desc, poolType, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        kernel, kernel, 0, 0, kernel, kernel));
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

    // grin
    cudnnTensorDescriptor_t grin_desc;
    cuCHECK(cudnnCreateTensorDescriptor(&grin_desc));
    cuCHECK(cudnnSetTensor4dDescriptor(grin_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));
    if (!isFirst)
        cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->grin_desc));
    gpuPrm->grin_desc = grin_desc;
      

    if (isFirst && !gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grin, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, insz.size() * sizeof(snFloat)));
    }
    else if (!gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_in));        gpuPrm->d_in = 0;
        cuCHECK(cudaFree(gpuPrm->d_out));       gpuPrm->d_out = 0;
        cuCHECK(cudaFree(gpuPrm->d_grin));      gpuPrm->d_grin = 0;
        cuCHECK(cudaFree(gpuPrm->d_grout));     gpuPrm->d_grout = 0;
     
        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grin, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, insz.size() * sizeof(snFloat)));
      }
}

void Pooling::freeParamCUDA(void* gpuPrms){
  
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;

    if (!gpuPrm) return;

    delete gpuPrm->cu_deviceProps;

    cuCHECK(cudnnDestroy(gpuPrm->cudnn));
    cuCHECK(cudnnDestroyPoolingDescriptor(gpuPrm->pool_desc));
    cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->in_desc));
    cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->out_desc));
    cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->grin_desc));
    cuCHECK(cudnnDestroyTensorDescriptor(gpuPrm->grout_desc));

    cudaFree(gpuPrm->d_in);
    cudaFree(gpuPrm->d_out);
    cudaFree(gpuPrm->d_grin);
    cudaFree(gpuPrm->d_grout);
}

void Pooling::forwardCUDA(poolType type, size_t kernel, const snSize& insz, snFloat* input,
    const snSize& outsz, snFloat* output, size_t* outputInx, void* gpuPrms){
  
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));       
    }

    // input
    cuCHECK(cudaMemcpy(gpuPrm->d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
      
    // run
    snFloat alpha = 1.f, beta = 0.f;
    cuCHECK(cudnnPoolingForward(gpuPrm->cudnn,
        gpuPrm->pool_desc,
        &alpha,
        gpuPrm->in_desc,
        gpuPrm->d_in,
        &beta,
        gpuPrm->out_desc,
        gpuPrm->d_out));
   
    // result
    cuCHECK(cudaMemcpy(output, gpuPrm->d_out, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_in)); 
        cuCHECK(cudaFree(gpuPrm->d_out));
    }
}

void Pooling::backwardCUDA(poolType type, size_t kernel, const snSize& outsz, size_t* outputInx, snFloat* output, snFloat* gradIn,
    const snSize& insz, snFloat* input, snFloat* gradOut, void* gpuPrms){
    
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grin, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_grout, insz.size() * sizeof(snFloat)));
    }

    // input
    cuCHECK(cudaMemcpy(gpuPrm->d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // output
    cuCHECK(cudaMemcpy(gpuPrm->d_out, output, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
   
    // grin
    cuCHECK(cudaMemcpy(gpuPrm->d_grin, gradIn, outsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));

    // run
    snFloat alpha = 1.f, beta = 0.f;
    cuCHECK(cudnnPoolingBackward(gpuPrm->cudnn,
        gpuPrm->pool_desc,
        &alpha,
        gpuPrm->out_desc,
        gpuPrm->d_out,
        gpuPrm->grin_desc,
        gpuPrm->d_grin,
        gpuPrm->in_desc,
        gpuPrm->d_in,
        &beta,
        gpuPrm->grout_desc,
        gpuPrm->d_grout));

    // result
    cuCHECK(cudaMemcpy(gradOut, gpuPrm->d_grout, insz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_in));
        cuCHECK(cudaFree(gpuPrm->d_out));
        cuCHECK(cudaFree(gpuPrm->d_grin));
        cuCHECK(cudaFree(gpuPrm->d_grout));
    }
}

#elif SN_CUDA

#include <cuda_runtime.h>
#include "../stdafx.h"
#include "SNOperator/src/Operator/pooling.h"

using namespace std;
using namespace SN_Base;

#ifndef cuCHECK
#define cuCHECK(func) if (func != 0){ ERROR_MESS("CUDA error: " + cudaGetErrorString(cudaGetLastError())); return;}
#endif

struct gpuParams{

    cudaDeviceProp* cu_deviceProps = 0;

    snFloat* d_in = 0;
    snFloat* d_out = 0;    
    size_t* d_idx = 0;
};

void Pooling::iniParamCUDA(const snSize& insz, const snSize& outsz, size_t kernel, void** pGpuPrm){
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
            cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm->d_idx, outsz.size() * sizeof(size_t)));
        }
    }
    else if (!gpuClearMem_){
                   
        cuCHECK(cudaFree(gpuPrm->d_in));  gpuPrm->d_in = 0;
        cuCHECK(cudaFree(gpuPrm->d_out)); gpuPrm->d_out = 0;
        cuCHECK(cudaFree(gpuPrm->d_idx)); gpuPrm->d_idx = 0;

        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_idx, outsz.size() * sizeof(size_t)));
    }
}

void Pooling::freeParamCUDA(void* gpuPrms){
    
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;

    if (!gpuPrm) return;

    delete gpuPrm->cu_deviceProps;

    cudaFree(gpuPrm->d_in);
    cudaFree(gpuPrm->d_out);
    cudaFree(gpuPrm->d_idx);  
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
    const snSize& outsz, snFloat* output, size_t* outputInx, void* gpuPrms){
   
    cudaSetDevice(gpuDeviceId_);
       
    gpuParams* gpuPrm = (gpuParams*)gpuPrms;
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&gpuPrm->d_in, insz.size() * sizeof(snFloat)));  
        cuCHECK(cudaMalloc(&gpuPrm->d_out, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm->d_idx, outsz.size() * sizeof(size_t))); 
    }

    // input
    cuCHECK(cudaMemcpy(gpuPrm->d_in, input, insz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
    
    // run     
    dim3 dimBlock(16, 16);
    dim3 dimGrid(unsigned int(outsz.d), unsigned int(outsz.n));

    cuPoolFwd <<< dimGrid, dimBlock >>>(type, kernel, insz, gpuPrm->d_in, outsz, gpuPrm->d_out, gpuPrm->d_idx);

    // result
    cuCHECK(cudaMemcpy(output, gpuPrm->d_out, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
    cuCHECK(cudaMemcpy(outputInx, gpuPrm->d_idx, outsz.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));

    if (gpuClearMem_){
        cuCHECK(cudaFree(gpuPrm->d_in));
        cuCHECK(cudaFree(gpuPrm->d_out));
        cuCHECK(cudaFree(gpuPrm->d_idx));
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

void Pooling::backwardCUDA(poolType type, size_t kernel, const snSize& outsz, size_t* outputInx, snFloat* output, snFloat* gradIn,
    const snSize& insz, SN_Base::snFloat* input, snFloat* gradOut, void* gpuPrms){
    cudaSetDevice(gpuDeviceId_);

    gpuParams* gpuPrm = (gpuParams*)gpuPrms;

    snFloat* d_grin = gpuPrm->d_out,
           * d_grout = gpuPrm->d_in;
    size_t* d_idx =  gpuPrm->d_idx;

    if (gpuClearMem_){
        cuCHECK(cudaMalloc(&d_grin, outsz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&d_grout, insz.size() * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&d_idx, outsz.size() * sizeof(size_t)));
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