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
#include "snOperator/src/Operator/deconvolution.h"
#include <omp.h>

using namespace std;
using namespace SN_Base;


void Deconvolution::forwardCPU(const deconvParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){
   
    size_t fWidth = prms.fWidth,
           fHeight = prms.fHeight,
           kernel = prms.kernel,
           stride = prms.stride,
           wStepByD = fWidth * fHeight,              
           wStepByK = wStepByD * kernel,             
           wStepByN = (wStepByK + 1) * insz.d,       
           inStepByD = insz.w * insz.h,              
           inStepByN = inStepByD * insz.d,           
           outStepByD = outsz.w * outsz.h,           
           outStepByN = outStepByD * outsz.d;        

    size_t shareStepByN = insz.d + outsz.d;          // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    memset(output, 0, outStepByN * outsz.n * sizeof(snFloat));

    // on batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* outBuff = share + insz.d + shareStepByN * n;

        for (size_t p = 0; p < inStepByD; ++p){

            size_t ix = p % insz.w, iy = p / insz.w,
                posW = ix * stride, posH = iy * stride;

            snFloat* pIn = input + ix + iy * insz.w + n * inStepByN;

            // on all input layers
            for (size_t d = 0; d < insz.d; ++d){
                inBuff[d] = *pIn;
                pIn += inStepByD;
            }

            // kernel
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pW = weight + cx + cy * fWidth;
                                
                memset(outBuff, 0, outsz.d * sizeof(snFloat));

                // on all input layers
                for (size_t d = 0; d < insz.d; ++d){

                    // on all output layers
                    snFloat in = inBuff[d];
                    for (size_t k = 0; k < kernel; ++k){
                        outBuff[k] += in * (*pW);
                        pW += wStepByD;
                    }
                    pW += 1;           // bias;
                }

                snFloat* pOut = output + (cx + posW) + (cy + posH) * outsz.w + n * outStepByN;

                for (size_t k = 0; k < kernel; ++k){
                    *pOut += outBuff[k];
                    pOut += outStepByD;
                }
            }
        }
    }

    free(share);


}

void Deconvolution::backwardCPU_GW(const deconvParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){
    
    size_t fWidth = prms.fWidth,
           fHeight = prms.fHeight,
           kernel = prms.kernel,
           stride = prms.stride,
           wStepByD = fWidth * fHeight,        
           wStepByK = wStepByD * kernel,       
           wStepByN = (wStepByK + 1) * insz.d, 
           inStepByD = insz.w * insz.h,        
           inStepByN = inStepByD * insz.d,     
           outStepByD = outsz.w * outsz.h,     
           outStepByN = outStepByD * outsz.d;  

    size_t shareStepByN = insz.d + outsz.d + insz.d;     // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    snFloat* wgThr = (insz.n == 1) ? dWeightOut : (snFloat*)calloc(wStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));
    memset(dWeightOut, 0, wStepByN * sizeof(snFloat));

    // по батчу
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* grinBuff = share + insz.d + shareStepByN * n;
        snFloat* groutBuff = share + insz.d + outsz.d + shareStepByN * n;
        snFloat* wBuff = wgThr + wStepByN * n;

        for (size_t p = 0; p < inStepByD; ++p){

            size_t ix = p % insz.w, iy = p / insz.w,
                posW = ix * stride, posH = iy * stride;

            snFloat* pIn = input + ix + iy * insz.w + n * inStepByN;
            snFloat* pdW = wBuff + wStepByK;

            // on all input layers
            for (size_t d = 0; d < insz.d; ++d){
                inBuff[d] = *pIn;   
                *(pdW + d) += *pIn;      // + bias

                pIn += inStepByD;
                pdW += wStepByK;               
            }

            memset(groutBuff, 0, insz.d * sizeof(snFloat));

            // kernel
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pGrIn = gradIn + (cx + posW) + (cy + posH) * outsz.w + n * outStepByN;
                snFloat* pW = weight + cx + cy * fWidth;
                snFloat* pdW = wBuff + cx + cy * fWidth;

                for (size_t k = 0; k < kernel; ++k){
                    grinBuff[k] = *pGrIn;
                    pGrIn += outStepByD;
                }

                // on all input layers
                for (size_t d = 0; d < insz.d; ++d){

                    // on all output layers
                    snFloat cin = inBuff[d], cout = 0;
                    for (size_t k = 0; k < kernel; ++k){
                        cout += grinBuff[k] * (*pW);
                        pW += wStepByD;

                        *pdW += grinBuff[k] * cin;
                        pdW += wStepByD;
                    }

                    pW += 1;             // bias
                    pdW += 1;         
                    groutBuff[d] += cout;
                }
            }

            snFloat* pOut = gradOut + ix + iy * insz.w + n * inStepByN;
            snFloat* pW = weight + wStepByK;

            // on all input layers
            for (size_t d = 0; d < insz.d; ++d){

                *pOut += groutBuff[d] + *(pW + d); // + bias (no change)

                pW += wStepByK;
                pOut += inStepByD;
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

void Deconvolution::backwardCPU_G(const deconvParams& prms,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut){

    size_t fWidth = prms.fWidth,
           fHeight = prms.fHeight,
           kernel = prms.kernel,
           stride = prms.stride,
           wStepByD = fWidth * fHeight,        
           wStepByK = wStepByD * kernel,       
           inStepByD = insz.w * insz.h,        
           inStepByN = inStepByD * insz.d,     
           outStepByD = outsz.w * outsz.h,     
           outStepByN = outStepByD * outsz.d;  

    size_t shareStepByN = outsz.d + insz.d;     // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));

    // on batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* grinBuff = share + shareStepByN * n;
        snFloat* groutBuff = share + outsz.d + shareStepByN * n;

        for (size_t p = 0; p < inStepByD; ++p){

            size_t ix = p % insz.w, iy = p / insz.w,
                posW = ix * stride, posH = iy * stride;

            memset(groutBuff, 0, insz.d * sizeof(snFloat));

            // kernel
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pGrIn = gradIn + (cx + posW) + (cy + posH) * outsz.w + n * outStepByN;
                snFloat* pW = weight + cx + cy * fWidth;

                for (size_t k = 0; k < kernel; ++k){
                    grinBuff[k] = *pGrIn;
                    pGrIn += outStepByD;
                }

                // on all input layers
                for (size_t d = 0; d < insz.d; ++d){

                    // on all output layers
                    snFloat cout = 0;
                    for (size_t k = 0; k < kernel; ++k){
                        cout += grinBuff[k] * (*pW);
                        pW += wStepByD;
                    }
                    pW += 1;           // bias;
                    groutBuff[d] += cout;
                }
            }

            snFloat* pOut = gradOut + ix + iy * insz.w + n * inStepByN;
            snFloat* pW = weight + wStepByK;

            // on all input layers
            for (size_t d = 0; d < insz.d; ++d){

                *pOut += groutBuff[d] + *(pW + d); // + bias (no change)

                pW += wStepByK;
                pOut += inStepByD;
            }
        }
    }

    free(share);
}


#ifndef SN_CUDA

void Deconvolution::iniParamCUDA(const snSize& insz, const snSize& outsz, const deconvParams& prms, void** gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

void Deconvolution::freeParamCUDA(void* gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

void Deconvolution::forwardCUDA(const deconvParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, void* gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

void Deconvolution::backwardCUDA_GW(const deconvParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, void* gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

void Deconvolution::backwardCUDA_G(const deconvParams& prms,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, void* gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

#endif

#ifndef SN_OpenCL

void Deconvolution::iniParamOCL(const snSize& insz, const snSize& outsz, const deconvParams& prms, void** gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void Deconvolution::freeParamOCL(void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void Deconvolution::forwardOCL(const deconvParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void Deconvolution::backwardOCL_GW(const deconvParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void Deconvolution::backwardOCL_G(const deconvParams& prms,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

#endif