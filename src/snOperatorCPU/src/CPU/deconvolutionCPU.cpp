//
// sunnet project
// Copyright (C) 2018 by Contributors <https://github.com/Tyill/sunnet>
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
#include "snOperatorCPU/src/Operator/deconvolution.h"
#include <thread>

using namespace std;
using namespace SN_Base;


void Deconvolution::forwardCPU(const deconvParams& prms,
    const snFloat* weight, const snSize& insz, const snFloat* input, const snSize& outsz, snFloat* output){
   
    size_t fWidth = prms.fWidth,
           fHeight = prms.fHeight,
           kernel = prms.kernel,
           stride = prms.stride,
           wStepByD = fWidth * fHeight,              // step weight by input
           inStepByD = insz.w * insz.h,              // step in by input
           inStepByN = inStepByD * insz.d,           // step in by batch
           outStepByD = outsz.w * outsz.h,           // step out by input
           outStepByN = outStepByD * outsz.d;        // step out by batch

    size_t shareStepByN = insz.d + outsz.d;          // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    memset(output, 0, outStepByN * outsz.n * sizeof(snFloat));

    auto core = std::thread::hardware_concurrency();
    if (core == 0) core = 4;

    // on batch
#pragma omp parallel for num_threads(core)
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* outBuff = share + insz.d + shareStepByN * n;

        for (size_t p = 0; p < inStepByD; ++p){

            size_t ix = p % insz.w, iy = p / insz.w,
                posW = ix * stride, posH = iy * stride;

            const snFloat* pIn = input + ix + iy * insz.w + n * inStepByN;

            // on all input layers
            for (size_t d = 0; d < insz.d; ++d){
                inBuff[d] = *pIn;
                pIn += inStepByD;
            }

            // kernel
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                const snFloat* pW = weight + cx + cy * fWidth;
                                
                memset(outBuff, 0, outsz.d * sizeof(snFloat));

                // on all input layers
                for (size_t d = 0; d < insz.d; ++d){

                    // on all output layers
                    snFloat in = inBuff[d];
                    for (size_t k = 0; k < kernel; ++k){
                        outBuff[k] += in * (*pW);
                        pW += wStepByD;
                    }                    
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
    const snFloat* weight, const snSize& insz, const snFloat* input, const snSize& outsz, const snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){
    
    size_t fWidth = prms.fWidth,
           fHeight = prms.fHeight,
           kernel = prms.kernel,
           stride = prms.stride,
           wStepByD = fWidth * fHeight,            // step weight by input
           wStepByK = wStepByD * kernel,           // step weight by output
           wStepByN = wStepByK * insz.d + insz.d,  // step weight by batch
           inStepByD = insz.w * insz.h,            // step in by input
           inStepByN = inStepByD * insz.d,         // step in by batch
           outStepByD = outsz.w * outsz.h,         // step out by input
           outStepByN = outStepByD * outsz.d;      // step out by batch

    size_t shareStepByN = insz.d + outsz.d + insz.d;     // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    snFloat* wgThr = (insz.n == 1) ? dWeightOut : (snFloat*)calloc(wStepByN * insz.n, sizeof(snFloat));

    memset(dWeightOut, 0, wStepByN * sizeof(snFloat));

    auto core = std::thread::hardware_concurrency();
    if (core == 0) core = 4;

    // РїРѕ Р±Р°С‚С‡Сѓ
#pragma omp parallel for num_threads(core)
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* grinBuff = share + insz.d + shareStepByN * n;
        snFloat* groutBuff = share + insz.d + outsz.d + shareStepByN * n;
        snFloat* wBuff = wgThr + wStepByN * n;

        for (size_t p = 0; p < inStepByD; ++p){

            size_t ix = p % insz.w, iy = p / insz.w,
                posW = ix * stride, posH = iy * stride;

            const snFloat* pIn = input + ix + iy * insz.w + n * inStepByN;
            snFloat* pdW = wBuff + wStepByK * insz.d;

            // on all input layers
            for (size_t d = 0; d < insz.d; ++d){
                inBuff[d] = *pIn;   
                *(pdW + d) += *pIn;      // + bias

                pIn += inStepByD;    
            }

            memset(groutBuff, 0, insz.d * sizeof(snFloat));

            // kernel
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                const snFloat* pGrIn = gradIn + (cx + posW) + (cy + posH) * outsz.w + n * outStepByN,
                             * pW = weight + cx + cy * fWidth;
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

                    groutBuff[d] += cout;
                }
            }

            snFloat* pOut = gradOut + ix + iy * insz.w + n * inStepByN;
            const snFloat* pW = weight + wStepByK * insz.d;

            // on all input layers
            for (size_t d = 0; d < insz.d; ++d){

                *pOut = groutBuff[d] + *(pW + d); // + bias (no change)

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
    const snFloat* weight, const snSize& insz, const snSize& outsz, const snFloat* gradIn, snFloat* gradOut){

    size_t fWidth = prms.fWidth,
           fHeight = prms.fHeight,
           kernel = prms.kernel,
           stride = prms.stride,                
           wStepByD = fWidth * fHeight,         // step weight by input
           wStepByK = wStepByD * kernel,        // step weight by output
           inStepByD = insz.w * insz.h,         // step in by input
           inStepByN = inStepByD * insz.d,      // step in by batch
           outStepByD = outsz.w * outsz.h,      // step out by input
           outStepByN = outStepByD * outsz.d;   // step out by batch

    size_t shareStepByN = outsz.d + insz.d;     // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));
  
    auto core = std::thread::hardware_concurrency();
    if (core == 0) core = 4;

    // on batch
#pragma omp parallel for num_threads(core)
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
                const snFloat* pGrIn = gradIn + (cx + posW) + (cy + posH) * outsz.w + n * outStepByN,
                             * pW = weight + cx + cy * fWidth;

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
                    
                    groutBuff[d] += cout;
                }
            }

            snFloat* pOut = gradOut + ix + iy * insz.w + n * inStepByN;
            const snFloat* pW = weight + wStepByK * insz.d;

            // on all input layers
            for (size_t d = 0; d < insz.d; ++d){

                *pOut = groutBuff[d] + *(pW + d); // + bias (no change)
                               
                pOut += inStepByD;
            }
        }
    }

    free(share);
}
