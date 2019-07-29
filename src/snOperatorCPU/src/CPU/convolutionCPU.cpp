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
#include "snOperatorCPU/src/Operator/convolution.h"
#include <thread>

#ifdef SN_AVX
#include "snSIMD/snSIMD.h"
#endif 

using namespace std;
using namespace SN_Base;



void Convolution::iniParamCPU(const snSize& insz, const snSize& outsz,
    const convParams& prms, CPUParams& cpuPrm){
     
#ifdef SN_AVX 
    // M * M * insz.d * outsz.w * outsz.h
    size_t sz = prms.fWidth * prms.fHeight * insz.d * outsz.w * outsz.h + 8; // 8 - end for _mm256_storeu_ps

    cpuPrm.buffMemFWD = (snFloat*)realloc(cpuPrm.buffMemFWD, sz * sizeof(snFloat));
#endif
}

void Convolution::freeParamCPU(CPUParams& cpuPrm){

    if (cpuPrm.buffMemFWD){

        free(cpuPrm.buffMemFWD);
    }
}

void forwardBASE(size_t kernel, size_t fWidth, size_t fHeight, size_t stride, size_t dilate,
    const snFloat* weight, const snSize& insz, const snFloat* input, const snSize& outsz, snFloat* output){

    size_t wStepByD = fWidth * fHeight,        // step weight by input
           wStepByK = wStepByD * insz.d,       // step weight by output
           wStepByN = wStepByK * kernel,       // step weight by batch
           inStepByD = insz.w * insz.h,        // step in by input
           inStepByN = inStepByD * insz.d,     // step in by batch
           outStepByD = outsz.w * outsz.h,     // step out by input
           outStepByN = outStepByD * outsz.d;  // step out by batch

    size_t shareStepByN = insz.d + kernel;     // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));
   
    auto core = std::thread::hardware_concurrency();
    if (core == 0) core = 4;

    // by batch
#pragma omp parallel for num_threads(core)
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
                const snFloat* pIn = input + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN,
                             * pW = weight + cx + cy * fWidth;

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
                    outBuff[k] += cout;
                }
            }

            snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
            const snFloat* pW = weight + wStepByN;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){

                *pOut = outBuff[k] + *(pW + k); // + bias              

                pOut += outStepByD;
            }
        }
    }

    free(share);
}

void Convolution::forwardCPU(const convParams& prms,
    const snFloat* weight, const snSize& insz, const snFloat* input, const snSize& outsz, snFloat* output, CPUParams& cpuPrm){
     
#ifdef SN_AVX   
        
    if ((prms.fWidth != prms.fHeight) || !SN_SIMD::convolutionFWD(prms.fWidth, prms.stride, prms.dilate,
                                             weight, insz, input, outsz, output, cpuPrm.buffMemFWD))

#endif
        forwardBASE(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate, weight, insz, input, outsz, output);
}

void backwardGW_BASE(size_t kernel, size_t fWidth, size_t fHeight, size_t stride, size_t dilate,
    const snFloat* weight, const snSize& insz, const snFloat* input, const snSize& outsz, const snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){

    size_t wStepByD = fWidth * fHeight,           // step weight by input
           wStepByK = wStepByD * insz.d,          // step weight by output
           wStepByN = wStepByK * kernel + kernel, // step weight by batch
           inStepByD = insz.w * insz.h,           // step in by input
           inStepByN = inStepByD * insz.d,        // step in by batch
           outStepByD = outsz.w * outsz.h,        // step out by input
           outStepByN = outStepByD * outsz.d;     // step out by batch

    size_t shareStepByN = insz.d + kernel + insz.d;      // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    snFloat* wgThr = (insz.n == 1) ? dWeightOut : (snFloat*)calloc(wStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));
    memset(dWeightOut, 0, wStepByN * sizeof(snFloat));

    auto core = std::thread::hardware_concurrency();
    if (core == 0) core = 4;

    // by batch
#pragma omp parallel for num_threads(core)
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* ginBuff = share + insz.d + shareStepByN * n;
        snFloat* goutBuff = share + insz.d + kernel + shareStepByN * n;
        snFloat* wBuff = wgThr + wStepByN * n;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            const snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;
            snFloat* pdW = wBuff + wStepByK * kernel;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){
                ginBuff[k] = *pGrIn;
                *(pdW + k) += *pGrIn;      // + bias

                pGrIn += outStepByD;
            }

            // kernel conv
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                const snFloat* pIn = input + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN,
                             * pW = weight + cx + cy * fWidth;
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

void Convolution::backwardCPU_GW(const convParams& prms,
    const snFloat* weight, const snSize& insz, const snFloat* input, const snSize& outsz, const snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){
    
#ifdef SN_AVX

    if ((prms.fWidth != prms.fHeight) || !SN_SIMD::convolutionBWD_GW(prms.fWidth, prms.stride, prms.dilate,
                                             weight, insz, input, outsz, gradIn, gradOut, dWeightOut))
#endif
        backwardGW_BASE(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate,
           weight, insz, input, outsz, gradIn, gradOut, dWeightOut);
}

void backwardG_Base(size_t kernel, size_t fWidth, size_t fHeight, size_t stride, size_t dilate,
    const snFloat* weight, const snSize& insz, const snSize& outsz, const snFloat* gradIn, snFloat* gradOut){

    size_t wStepByD = fWidth * fHeight,         // step weight by input         
           inStepByD = insz.w * insz.h,         // step in by input
           inStepByN = inStepByD * insz.d,      // step in by batch
           outStepByD = outsz.w * outsz.h,      // step out by input
           outStepByN = outStepByD * outsz.d;   // step out by batch

    size_t shareStepByN = kernel + insz.d;          // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));

    auto core = std::thread::hardware_concurrency();
    if (core == 0) core = 4;

    // by batch
#pragma omp parallel for num_threads(core)
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* ginBuff = share + shareStepByN * n;
        snFloat* goutBuff = share + kernel + shareStepByN * n;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            const snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){
                ginBuff[k] = *pGrIn;
                pGrIn += outStepByD;
            }

            // kernel conv
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                const snFloat* pW = weight + cx + cy * fWidth;
                              
                memset(goutBuff, 0, insz.d * sizeof(snFloat));

                // on all out layers
                for (size_t k = 0; k < kernel; ++k){

                    // on all in layers
                    snFloat gin = ginBuff[k];
                    for (size_t d = 0; d < insz.d; ++d){
                        goutBuff[d] += gin * (*pW);
                        pW += wStepByD;
                    }
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

void Convolution::backwardCPU_G(const convParams& prms,
    const snFloat* weight, const snSize& insz, const snSize& outsz, const snFloat* gradIn, snFloat* gradOut){

#ifdef SN_AVX

    if ((prms.fWidth != prms.fHeight) || !SN_SIMD::convolutionBWD_G(prms.fWidth, prms.stride, prms.dilate,
                                                weight, insz, outsz, gradIn, gradOut))
#endif
    backwardG_Base(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate,
        weight, insz, outsz, gradIn, gradOut);

}