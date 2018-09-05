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
#include "SNOperator/src/Operator/convolution.h"

using namespace std;
using namespace SN_Base;
          
texture<snFloat, cudaTextureType2DLayered> cu_texIn;

__global__ void cuConvolutionFwd(size_t fWidth, size_t fHeight, size_t stride,
    snFloat* weight, snSize insz, snSize outsz, snFloat* output){

    size_t wStepByD = fWidth * fHeight,        // шаг весов по входу
           wStepByK = wStepByD * insz.d + 1,   // шаг весов по выходу
           outStepByD = outsz.w * outsz.h;     // шаг вых слоя по выходу
          
    // gridDim.x - кол-во вых слоев
    // gridDim.y - размер батча

    weight += blockIdx.x * wStepByK;

    extern __shared__ snFloat share[];

    if (threadIdx.x == 0){           
       
        // буфер иниц-я bias
        memset(share, weight[wStepByD * insz.d], outStepByD * sizeof(snFloat));

        // копирование весов в разделяемую память, если хватает памяти
        if (wStepByK < 16000)  // 16 * 4(float) = 64kb
            memcpy(share + outStepByD, weight, wStepByK * sizeof(snFloat)); 
    }
    __syncthreads();

    if (wStepByK < 16000)  // 16 * 4(float) = 64kb
        weight = share + outStepByD;
  
    unsigned int oz = threadIdx.z;
    while (oz < insz.d){

       unsigned int oy = threadIdx.y;
       while (oy < outsz.h){

           unsigned int ox = threadIdx.x;
           while (ox < outsz.w){

                size_t posW = ox * stride, posH = oy * stride;

                snFloat* w = weight + oz * wStepByD;
                
                // ядро свертки   
                snFloat csum = 0; 
                for (size_t c = 0; c < wStepByD; ++c){

                    size_t cx = c % fWidth, cy = c / fWidth;
                 
                    snFloat u = (cx + posW + 0.5f) / (snFloat)insz.w;
                    snFloat v = (cy + posH + 0.5f) / (snFloat)insz.h;
                    snFloat d = (oz + blockIdx.y * insz.d + 0.5f) / (snFloat)(insz.d * insz.n);
                   
                    csum += tex2DLayered(cu_texIn, u, v, d) * w[cx + cy * fWidth];
                }
                atomicAdd(&(share[ox + oy * outsz.w]), csum);
               
                ox += blockDim.x; 
            }
            oy += blockDim.y; 
        }
        oz += blockDim.z; 
    }    
    __syncthreads();

    // на выход
    if (threadIdx.x == 0){

        snFloat* out = output + blockIdx.x * blockIdx.y * outStepByD;
        
        memcpy(out, share, outStepByD * sizeof(snFloat));
    }  
}

void Convolution::iniParamCUDA(snSize insz, snSize outsz, size_t fWidth, size_t fHeight, map<string, snFloat*>& gpuPrm){

    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, 0);
    if (deviceProps.major < 2){
        ERROR_MESS("%s requires SM >= 2.0 to support Texture Arrays.  Test will be waived... \n");
        return;
    }

    // set texture parameters
    cu_texIn.addressMode[0] = cudaAddressModeWrap;
    cu_texIn.addressMode[1] = cudaAddressModeWrap;
    cu_texIn.filterMode = cudaFilterModeLinear;
    cu_texIn.normalized = true;  // access with normalized texture coordinates
           
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaArray_t d_in_fwd = nullptr;
    cudaMalloc3DArray(&d_in_fwd, &channelDesc, make_cudaExtent(insz.w, insz.h, insz.d * insz.n), cudaArrayLayered);
    gpuPrm["d_in_fwd"] = (snFloat*)d_in_fwd;

    snFloat *d_w_fwd = nullptr;
    cudaMalloc((void **)&d_w_fwd, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat));
    gpuPrm["d_w_fwd"] = (snFloat*)d_w_fwd;

    snFloat *d_out_fwd = nullptr;
    cudaMalloc((void **)&d_out_fwd, outsz.w * outsz.h * outsz.d * outsz.n * sizeof(snFloat));
    gpuPrm["d_out_fwd"] = d_out_fwd;
}

void Convolution::freeParamCUDA(map<std::string, snFloat*>& gpuPrm){

    for (auto p : gpuPrm){
        if (p.first == "d_in_fwd") 
            cudaFreeArray((cudaArray_t)p.second);
        else
            cudaFree(p.second); 
    }    
}

void Convolution::forwardCUDA(size_t kernel, size_t fWidth, size_t fHeight, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* output, map<string, snFloat*>& gpuPrm){
  
    // вход данные
    cudaArray_t d_inArray = (cudaArray_t)gpuPrm["d_in_fwd"];

    cudaMemcpy3DParms prms = { 0 };
    prms.srcPos = make_cudaPos(0, 0, 0);
    prms.dstPos = make_cudaPos(0, 0, 0);
    prms.srcPtr = make_cudaPitchedPtr(input, insz.w * sizeof(snFloat), insz.w, insz.h);
    prms.dstArray = d_inArray;
    prms.extent = make_cudaExtent(insz.w, insz.h, insz.d * insz.n);
    prms.kind = cudaMemcpyHostToDevice;
    cudaError_t err = cudaMemcpy3D(&prms);

    // привязка текстуры
    err = cudaBindTextureToArray(&cu_texIn, d_inArray, &cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat));

    // веса
    snFloat* d_w = gpuPrm["d_w_fwd"];
    err = cudaMemcpy(d_w, weight, (fWidth * fHeight * insz.d + 1) * outsz.d * sizeof(snFloat), cudaMemcpyHostToDevice);

    // выход
    snFloat* d_out = gpuPrm["d_out_fwd"];
   
    // выполнение
    dim3 dimBlock(8, 8);
    dim3 dimGrid(outsz.d, outsz.n);

    size_t shareSz = outsz.w * outsz.h + fWidth * fHeight * insz.d + 1;
    cuConvolutionFwd <<< dimGrid, dimBlock, shareSz * sizeof(snFloat) >>>(fWidth, fHeight, stride,
        d_w, insz, outsz, d_out);
     
   // err = cudaGetLastError();

    err = cudaUnbindTexture(&cu_texIn);
 
    // результ
    err = cudaMemcpy(output, d_out, outsz.w * outsz.h * outsz.d * outsz.n * sizeof(snFloat), cudaMemcpyDeviceToHost);

    bool ff = false;
}

void Convolution::backwardCUDA_GW(size_t kernel, size_t fWidth, size_t fHeight, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, map<string, snFloat*>&){

    size_t wStepByD = fWidth * fHeight,                  // шаг весов по входу
        wStepByK = wStepByD * insz.d,                 // шаг весов по выходу
        wStepByN = (wStepByK + 1) * kernel,           // шаг весов по батчу
        inStepByD = insz.w * insz.h,                  // шаг вх слоя по входу
        inStepByN = inStepByD * insz.d,               // шаг вх слоя по батчу
        outStepByD = outsz.w * outsz.h,               // шаг вых слоя по выходу
        outStepByN = outStepByD * outsz.d;            // шаг вых слоя по батчу

    size_t shareStepByN = insz.d + kernel + insz.d;      // для локализации памяти
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    snFloat* wgThr = (insz.n == 1) ? dWeightOut : (snFloat*)calloc(wStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));
    memset(dWeightOut, 0, wStepByN * sizeof(snFloat));

    // по батчу  
#pragma omp parallel for
    for (int n = 0; n < insz.n; ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* ginBuff = share + insz.d + shareStepByN * n;
        snFloat* goutBuff = share + insz.d + kernel + shareStepByN * n;
        snFloat* wBuff = wgThr + wStepByN * n;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;
            snFloat* pdW = wBuff + wStepByK;

            // по всем вых слоям
            for (size_t k = 0; k < kernel; ++k){
                ginBuff[k] = *pGrIn;

                *(pdW + k) += *pGrIn;      // + bias

                pGrIn += outStepByD;
                pdW += wStepByK;
            }

            // ядро свертки
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pIn = input + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;
                snFloat* pW = weight + cx + cy * fWidth;
                snFloat* pdW = wBuff + cx + cy * fWidth;

                for (size_t d = 0; d < insz.d; ++d){
                    inBuff[d] = *pIn;
                    pIn += inStepByD;
                }

                memset(goutBuff, 0, insz.d * sizeof(snFloat));

                // по всем вых слоям
                for (size_t k = 0; k < kernel; ++k){

                    // по всем вх слоям
                    snFloat gin = ginBuff[k];
                    for (size_t d = 0; d < insz.d; ++d){
                        goutBuff[d] += gin * (*pW);
                        pW += wStepByD;

                        *pdW += gin * inBuff[d];
                        pdW += wStepByD;
                    }
                    pW += 1;           // bias;
                    pdW += 1;
                }

                snFloat* pGrOut = gradOut + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;

                for (size_t d = 0; d < insz.d; ++d){
                    *pGrOut += goutBuff[d];
                    pGrOut += inStepByD;
                }
            }
        }
    }

    if (insz.n > 1){
        for (size_t i = 0; i < insz.n; ++i){
            snFloat* wBuff = wgThr + wStepByN * i;
            for (size_t j = 0; j < wStepByN; ++j)
                dWeightOut[j] += wBuff[j];
        }
        for (size_t j = 0; j < wStepByN; ++j)
            dWeightOut[j] /= insz.n;

        free(wgThr);
    }

    free(share);
}

void Convolution::backwardCUDA_G(size_t kernel, size_t fWidth, size_t fHeight, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* gradIn, snFloat* gradOut, map<string, snFloat*>&){

    size_t wStepByD = fWidth * fHeight,                  // шаг весов по входу
        inStepByD = insz.w * insz.h,                  // шаг вх слоя по входу
        inStepByN = inStepByD * insz.d,               // шаг вх слоя по батчу
        outStepByD = outsz.w * outsz.h,               // шаг вых слоя по выходу
        outStepByN = outStepByD * outsz.d;            // шаг вых слоя по батчу

    size_t shareStepByN = insz.d + kernel + insz.d;          // для локализации памяти
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));

    // по батчу  
#pragma omp parallel for
    for (int n = 0; n < insz.n; ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* ginBuff = share + insz.d + shareStepByN * n;
        snFloat* goutBuff = share + insz.d + kernel + shareStepByN * n;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;

            // по всем вых слоям
            for (size_t k = 0; k < kernel; ++k){
                ginBuff[k] = *pGrIn;
                pGrIn += outStepByD;
            }

            // ядро свертки
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pIn = input + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;
                snFloat* pW = weight + cx + cy * fWidth;

                for (size_t d = 0; d < insz.d; ++d){
                    inBuff[d] = *pIn;
                    pIn += inStepByD;
                }

                memset(goutBuff, 0, insz.d * sizeof(snFloat));

                // по всем вых слоям
                for (size_t k = 0; k < kernel; ++k){

                    // по всем вх слоям
                    snFloat gin = ginBuff[k];
                    for (size_t d = 0; d < insz.d; ++d){
                        goutBuff[d] += gin * (*pW);
                        pW += wStepByD;
                    }
                    pW += 1;           // bias;
                }

                snFloat* pGrOut = gradOut + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;

                for (size_t d = 0; d < insz.d; ++d){
                    *pGrOut += goutBuff[d];
                    pGrOut += inStepByD;
                }
            }
        }
    }

    free(share);
}

#endif 
