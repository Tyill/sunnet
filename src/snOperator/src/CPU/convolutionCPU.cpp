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
#include "snOperator/src/Operator/convolution.h"
#include <omp.h>

using namespace std;
using namespace SN_Base;


void Convolution::forwardCPU(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){
   
    size_t kernel = prms.kernel,
           fWidth = prms.fWidth,
           fHeight = prms.fHeight,
           stride = prms.stride,
           dilate = prms.dilate,
           wStepByD = fWidth * fHeight,        // step weight by input
           wStepByK = wStepByD * insz.d,       // step weight by output
           inStepByD = insz.w * insz.h,        // step in by input
           inStepByN = inStepByD * insz.d,     // step in by batch
           outStepByD = outsz.w * outsz.h,     // step out by input
           outStepByN = outStepByD * outsz.d;  // step out by batch

    size_t shareStepByN = insz.d + kernel;     // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));
    
    memset(output, 0, outStepByN * insz.n * sizeof(snFloat));
        
    // by batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* outBuff = share + insz.d + shareStepByN * n;
        
        for (size_t p = 0; p < outStepByD; ++p){
        
            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            memset(outBuff, 0, kernel * sizeof(snFloat));

            // kernel conv
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pIn = input + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN;
                snFloat* pW = weight + cx + cy * fWidth;

                for (size_t d = 0; d < insz.d; ++d){
                    inBuff[d] = *pIn;
                    pIn += inStepByD;
                }
                            
                // on all out layers
                for (size_t k = 0; k < kernel; ++k){
                                        
                    // on all in layers
                    snFloat cout = 0;
                    for (size_t d = 0; d < insz.d; ++d){
                        cout += inBuff[d] * (*pW);
                        pW += wStepByD;
                    }
                    pW += 1;           // bias;
                    outBuff[k] += cout;
                }
            }

            snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
            snFloat* pW = weight + wStepByK;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){
               
                *pOut += outBuff[k] + *(pW + k); // + bias
               
                pW += wStepByK;
                pOut += outStepByD;
            }
        }        
    }

    free(share); 
}

void Convolution::backwardCPU_GW(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){
    
    size_t kernel = prms.kernel,
           fWidth = prms.fWidth,
           fHeight = prms.fHeight,
           stride = prms.stride,
           dilate = prms.dilate,
           wStepByD = fWidth * fHeight,         // step weight by input
           wStepByK = wStepByD * insz.d,        // step weight by output
           wStepByN = (wStepByK + 1) * kernel,  // step weight by batch
           inStepByD = insz.w * insz.h,         // step in by input
           inStepByN = inStepByD * insz.d,      // step in by batch
           outStepByD = outsz.w * outsz.h,      // step out by input
           outStepByN = outStepByD * outsz.d;   // step out by batch

    size_t shareStepByN = insz.d + kernel + insz.d;      // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    snFloat* wgThr = (insz.n == 1) ? dWeightOut : (snFloat*)calloc(wStepByN * insz.n, sizeof(snFloat));
    
    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));
    memset(dWeightOut, 0, wStepByN * sizeof(snFloat));
    
    // by batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* ginBuff = share + insz.d + shareStepByN * n;
        snFloat* goutBuff = share + insz.d + kernel + shareStepByN * n;
        snFloat* wBuff = wgThr + wStepByN * n;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;
            snFloat* pdW = wBuff + wStepByK;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){
                ginBuff[k] = *pGrIn;

                *(pdW + k) += *pGrIn;      // + bias

                pGrIn += outStepByD;
                pdW += wStepByK;
            }

            // kernel conv
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pIn = input + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN;
                snFloat* pW = weight + cx + cy * fWidth;
                snFloat* pdW = wBuff + cx + cy * fWidth;

                for (size_t d = 0; d < insz.d; ++d){
                    inBuff[d] = *pIn;
                    pIn += inStepByD;
                }

                memset(goutBuff, 0, insz.d * sizeof(snFloat));

                // on all out layers
                for (size_t k = 0; k < kernel; ++k){

                    // on all in layers
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

                snFloat* pGrOut = gradOut + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN;

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

void Convolution::backwardCPU_G(const convParams& prms,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut){

    size_t kernel = prms.kernel,
           fWidth = prms.fWidth,
           fHeight = prms.fHeight,
           stride = prms.stride,
           dilate = prms.dilate,
           wStepByD = fWidth * fHeight,         // step weight by input
           wStepByK = wStepByD * insz.d,        // step weight by output
           wStepByN = (wStepByK + 1) * kernel,  // step weight by batch
           inStepByD = insz.w * insz.h,         // step in by input
           inStepByN = inStepByD * insz.d,      // step in by batch
           outStepByD = outsz.w * outsz.h,      // step out by input
           outStepByN = outStepByD * outsz.d;   // step out by batch

    size_t shareStepByN = kernel + insz.d;          // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));

    // by batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* ginBuff = share + shareStepByN * n;
        snFloat* goutBuff = share + kernel + shareStepByN * n;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){
                ginBuff[k] = *pGrIn;
                pGrIn += outStepByD;
            }

            // kernel conv
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pW = weight + cx + cy * fWidth;
                              
                memset(goutBuff, 0, insz.d * sizeof(snFloat));

                // on all out layers
                for (size_t k = 0; k < kernel; ++k){

                    // on all in layers
                    snFloat gin = ginBuff[k];
                    for (size_t d = 0; d < insz.d; ++d){
                        goutBuff[d] += gin * (*pW);
                        pW += wStepByD;
                    }
                    pW += 1;           // bias;
                }

                snFloat* pGrOut = gradOut + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN;

                for (size_t d = 0; d < insz.d; ++d){
                    *pGrOut += goutBuff[d];
                    pGrOut += inStepByD;
                }
            }
        }
    }

    free(share);
}


#ifndef SN_CUDA

/// init aux params CUDA          
void Convolution::iniParamCUDA(const SN_Base::snSize& insz, const SN_Base::snSize& outsz,
    const convParams&, void** gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

/// free aux params CUDA          
void Convolution::freeParamCUDA(void* gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

void Convolution::forwardCUDA(const convParams&,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, void* gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

void Convolution::backwardCUDA_GW(const convParams&,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, void* gpuPrm){
    ERROR_MESS("CUDA non compiler");

}

void Convolution::backwardCUDA_G(const convParams&,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, void* gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

#endif

#ifndef SN_OpenCL

/// init aux params OpenCL          
void Convolution::iniParamOCL(const snSize& insz, const snSize& outsz,
    const convParams&, void** gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

/// free aux params OpenCL           
void Convolution::freeParamOCL(void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void Convolution::forwardOCL(const convParams&,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void Convolution::backwardOCL_GW(const convParams&,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");

}

void Convolution::backwardOCL_G(const convParams&,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

#endif