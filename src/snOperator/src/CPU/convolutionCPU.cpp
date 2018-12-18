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

#ifdef SN_AVX
#include <immintrin.h>

void forwardAVX_K3(size_t kernel, size_t stride, size_t dilate,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

    size_t wStepByD = 9,                       // step weight by input
        wStepByK = wStepByD * insz.d,       // step weight by output
        wStepByN = wStepByK * kernel,       // step weight by batch
        inStepByD = insz.w * insz.h,        // step in by input
        inStepByN = inStepByD * insz.d,     // step in by batch
        outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch

    size_t shareStepByN = kernel;     // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    memset(output, 0, outStepByN * insz.n * sizeof(snFloat));

    // by batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* outBuff = share + shareStepByN * n;
        __m256 arIn, arW, arOut;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            memset(outBuff, 0, kernel * sizeof(snFloat));

            snFloat* pW = weight;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){

                snFloat* pIn = input + n * inStepByN;

                arOut = _mm256_setzero_ps();

                // on all in layers
                for (size_t d = 0; d < insz.d; ++d){

#define dIn(c, r) *(pIn + ((c) + posW + (c) * (dilate - 1)) + ((r) + posH + (r) * (dilate - 1)) * insz.w)

                    // kernel conv
                    arIn = _mm256_setr_ps(dIn(0, 0), dIn(1, 0), dIn(2, 0),
                                          dIn(0, 1), dIn(1, 1), dIn(2, 1),
                                          dIn(0, 2), dIn(1, 2));
#define WN 3
#define dW(c, r) *(pW + (c) + (r) * WN)

                    arW = _mm256_setr_ps(dW(0, 0), dW(1, 0), dW(2, 0),
                                         dW(0, 1), dW(1, 1), dW(2, 1),
                                         dW(0, 2), dW(1, 2));

                    arOut = _mm256_add_ps(arOut, _mm256_mul_ps(arIn, arW));

                    outBuff[k] += dIn(WN - 1, WN - 1) * dW(WN - 1, WN - 1);

                    pIn += inStepByD;
                    pW += wStepByD;
                }

                for (size_t c = 0; c < 8; ++c)
                    outBuff[k] += arOut.m256_f32[c];
            }

            snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
            pW = weight + wStepByN;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){

                *pOut += outBuff[k] + *(pW + k); // + bias              

                pOut += outStepByD;
            }
        }
    }

#undef dIn
#undef WN
#undef dW

    free(share);
}

void forwardAVX_K5(size_t kernel, size_t stride, size_t dilate,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

    size_t wStepByD = 25,                   // step weight by input
        wStepByK = wStepByD * insz.d,       // step weight by output
        wStepByN = wStepByK * kernel,       // step weight by batch
        inStepByD = insz.w * insz.h,        // step in by input
        inStepByN = inStepByD * insz.d,     // step in by batch
        outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch

    size_t shareStepByN = kernel;     // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    memset(output, 0, outStepByN * insz.n * sizeof(snFloat));

    // by batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* outBuff = share + shareStepByN * n;
        __m256 arIn, arW, arOut[3];

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            memset(outBuff, 0, kernel * sizeof(snFloat));

            snFloat* pW = weight;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){

                snFloat* pIn = input + n * inStepByN;

                arOut[0] = _mm256_setzero_ps();

                // on all in layers
                for (size_t d = 0; d < insz.d; ++d){

#define WN 5
#define dIn(c, r) *(pIn + ((c) + posW + (c) * (dilate - 1)) + ((r) + posH + (r) * (dilate - 1)) * insz.w)
#define dW(c, r) *(pW + (c) + (r) * WN)


                    outBuff[k] += dIn(WN - 1, WN - 1) * dW(WN - 1, WN - 1);

                    pIn += inStepByD;
                    pW += wStepByD;
                }

                for (size_t c = 0; c < 8; ++c)
                    outBuff[k] += arOut[0].m256_f32[c];
            }

            snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
            pW = weight + wStepByN;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){

                *pOut += outBuff[k] + *(pW + k); // + bias              

                pOut += outStepByD;
            }
        }
    }

#undef dIn
#undef WN
#undef dW

    free(share);
}

void backwardGW_AVX_K3(size_t kernel, size_t stride, size_t dilate,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){

    size_t wStepByD = 9,                          // step weight by input
        wStepByK = wStepByD * insz.d,          // step weight by output
        wStepByN = wStepByK * kernel + kernel, // step weight by batch
        inStepByD = insz.w * insz.h,           // step in by input
        inStepByN = inStepByD * insz.d,        // step in by batch
        outStepByD = outsz.w * outsz.h,        // step out by input
        outStepByN = outStepByD * outsz.d;     // step out by batch

    snFloat* wgThr = (insz.n == 1) ? dWeightOut : (snFloat*)calloc(wStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));
    memset(dWeightOut, 0, wStepByN * sizeof(snFloat));


    // by batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* wBuff = wgThr + wStepByN * n;
        __m256 arIn, arGIn, arW, arDW, arGOut;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;
            snFloat* pdW = wBuff + wStepByK * kernel;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){
                *(pdW + k) += pGrIn[k * outStepByD];      // + bias
            }

            snFloat* pGrOut = gradOut + n * inStepByN;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){

                snFloat* pIn = input + n * inStepByN;
                pGrOut = gradOut + n * inStepByN;

                snFloat gin = pGrIn[k * outStepByD];
                snFloat* pW = weight;
                pdW = wBuff;

                // on all in layers               
                for (size_t d = 0; d < insz.d; ++d){
#define WN 3
#define dIn(c, r) *(pIn + ((c) + posW + (c) * (dilate - 1)) + ((r) + posH + (r) * (dilate - 1)) * insz.w)

                    arIn = _mm256_setr_ps(dIn(0, 0), dIn(1, 0), dIn(2, 0),
                        dIn(0, 1), dIn(1, 1), dIn(2, 1),
                        dIn(0, 2), dIn(1, 2));

                    arGIn = _mm256_set1_ps(gin);

#define dW(c, r) *(pW + (c) + (r) * WN)

                    arW = _mm256_setr_ps(dW(0, 0), dW(1, 0), dW(2, 0),
                        dW(0, 1), dW(1, 1), dW(2, 1),
                        dW(0, 2), dW(1, 2));

#define dDW(c, r) *(pdW + (c) + (r) * WN)

                    arDW = _mm256_setr_ps(dDW(0, 0), dDW(1, 0), dDW(2, 0),
                        dDW(0, 1), dDW(1, 1), dDW(2, 1),
                        dDW(0, 2), dDW(1, 2));

                    arGOut = _mm256_add_ps(arGOut, _mm256_mul_ps(arGIn, arW));
                    arDW = _mm256_add_ps(arDW, _mm256_mul_ps(arGIn, arIn));

#define dGOut(c, r) *(pGrOut + ((c) + posW + (c) * (dilate - 1)) + ((r) + posH + (r) * (dilate - 1)) * insz.w)

                    dGOut(WN - 1, WN - 1) += gin * dW(WN - 1, WN - 1);
                    dDW(WN - 1, WN - 1) += gin * dIn(WN - 1, WN - 1);

                    pIn += inStepByD;
                    pGrOut += inStepByD;
                    pW += wStepByD;
                    pdW += wStepByD;
                }
            }

            pGrOut = gradOut + n * inStepByN;
            pdW = wBuff;

            for (size_t d = 0; d < insz.d; ++d){

                for (size_t r = 0; r < WN; ++r){
                    for (size_t c = 0; c < WN; ++c){

                        if ((r < WN - 1) || (c < WN - 1)){
                            dGOut(c, r) += arGOut.m256_f32[r * WN + c];
                            dDW(c, r) += arDW.m256_f32[r * WN + c];
                        }
                    }
                }

                pGrOut += inStepByD;
                pdW += wStepByD;
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

#undef dIn
#undef WN
#undef dGOut
#undef dDW
#undef dW

}

void backwardG_AVX_K3(size_t kernel, size_t stride, size_t dilate,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut){

//    size_t wStepByD = 9,                          // step weight by input
//        wStepByK = wStepByD * insz.d,          // step weight by output
//        wStepByN = wStepByK * kernel + kernel, // step weight by batch
//        inStepByD = insz.w * insz.h,           // step in by input
//        inStepByN = inStepByD * insz.d,        // step in by batch
//        outStepByD = outsz.w * outsz.h,        // step out by input
//        outStepByN = outStepByD * outsz.d;     // step out by batch
//
//    snFloat* wgThr = (insz.n == 1) ? dWeightOut : (snFloat*)calloc(wStepByN * insz.n, sizeof(snFloat));
//
//    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));
//    memset(dWeightOut, 0, wStepByN * sizeof(snFloat));
//
//
//    // by batch
//#pragma omp parallel for
//    for (int n = 0; n < int(insz.n); ++n){
//
//        snFloat* wBuff = wgThr + wStepByN * n;
//        __m256 arIn, arGIn, arW, arDW, arGOut;
//
//        for (size_t p = 0; p < outStepByD; ++p){
//
//            size_t ox = p % outsz.w, oy = p / outsz.w,
//                posW = ox * stride, posH = oy * stride;
//
//            snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;
//            snFloat* pdW = wBuff + wStepByK * kernel;
//
//            // on all out layers
//            for (size_t k = 0; k < kernel; ++k){
//                *(pdW + k) += pGrIn[k * outStepByD];      // + bias
//            }
//
//            snFloat* pGrOut = gradOut + n * inStepByN;
//
//            // on all out layers
//            for (size_t k = 0; k < kernel; ++k){
//
//                snFloat* pIn = input + n * inStepByN;
//                pGrOut = gradOut + n * inStepByN;
//
//                snFloat gin = pGrIn[k * outStepByD];
//                snFloat* pW = weight;
//                pdW = wBuff;
//
//                // on all in layers               
//                for (size_t d = 0; d < insz.d; ++d){
//#define WN 3
//#define dIn(c, r) *(pIn + ((c) + posW + (c) * (dilate - 1)) + ((r) + posH + (r) * (dilate - 1)) * insz.w)
//
//                    arIn = _mm256_setr_ps(dIn(0, 0), dIn(1, 0), dIn(2, 0),
//                        dIn(0, 1), dIn(1, 1), dIn(2, 1),
//                        dIn(0, 2), dIn(1, 2));
//
//                    arGIn = _mm256_set1_ps(gin);
//
//#define dW(c, r) *(pW + (c) + (r) * WN)
//
//                    arW = _mm256_setr_ps(dW(0, 0), dW(1, 0), dW(2, 0),
//                        dW(0, 1), dW(1, 1), dW(2, 1),
//                        dW(0, 2), dW(1, 2));
//
//#define dDW(c, r) *(pdW + (c) + (r) * WN)
//
//                    arDW = _mm256_setr_ps(dDW(0, 0), dDW(1, 0), dDW(2, 0),
//                        dDW(0, 1), dDW(1, 1), dDW(2, 1),
//                        dDW(0, 2), dDW(1, 2));
//
//                    arGOut = _mm256_add_ps(arGOut, _mm256_mul_ps(arGIn, arW));
//                    arDW = _mm256_add_ps(arDW, _mm256_mul_ps(arGIn, arIn));
//
//#define dGOut(c, r) *(pGrOut + ((c) + posW + (c) * (dilate - 1)) + ((r) + posH + (r) * (dilate - 1)) * insz.w)
//
//                    dGOut(WN - 1, WN - 1) += gin * dW(WN - 1, WN - 1);
//                    dDW(WN - 1, WN - 1) += gin * dIn(WN - 1, WN - 1);
//
//                    pIn += inStepByD;
//                    pGrOut += inStepByD;
//                    pW += wStepByD;
//                    pdW += wStepByD;
//                }
//            }
//
//            pGrOut = gradOut + n * inStepByN;
//            pdW = wBuff;
//
//            for (size_t d = 0; d < insz.d; ++d){
//
//                for (size_t r = 0; r < WN; ++r){
//                    for (size_t c = 0; c < WN; ++c){
//
//                        if ((r < WN - 1) || (c < WN - 1)){
//                            dGOut(c, r) += arGOut.m256_f32[r * WN + c];
//                            dDW(c, r) += arDW.m256_f32[r * WN + c];
//                        }
//                    }
//                }
//
//                pGrOut += inStepByD;
//                pdW += wStepByD;
//            }
//        }
//    }
//
//    if (insz.n > 1){
//        for (size_t i = 0; i < insz.n; ++i){
//            snFloat* wBuff = wgThr + wStepByN * i;
//            for (size_t j = 0; j < wStepByN; ++j)
//                dWeightOut[j] += wBuff[j];
//        }
//        for (size_t j = 0; j < wStepByN; ++j)
//            dWeightOut[j] /= insz.n;
//
//        free(wgThr);
//    }
//
//#undef dIn
//#undef WN
//#undef dGOut
//#undef dDW
//#undef dW

}

#endif


void forwardBASE(size_t kernel, size_t fWidth, size_t fHeight, size_t stride, size_t dilate,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

    size_t wStepByD = fWidth * fHeight,        // step weight by input
           wStepByK = wStepByD * insz.d,       // step weight by output
           wStepByN = wStepByK * kernel,       // step weight by batch
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
                    outBuff[k] += cout;
                }
            }

            snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
            snFloat* pW = weight + wStepByN;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){

                *pOut += outBuff[k] + *(pW + k); // + bias              

                pOut += outStepByD;
            }
        }
    }

    free(share);
}

void Convolution::forwardCPU(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){
      
#ifdef SN_AVX
   
    if ((prms.fWidth == 3) && (prms.fHeight == 3))
        forwardAVX_K3(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);
  /*  else if ((prms.fWidth == 5) && (prms.fHeight == 5))
        forwardAVX_K5(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);
    else if ((prms.fWidth == 7) && (prms.fHeight == 7))
        forwardAVX_K7(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);
    else if ((prms.fWidth == 9) && (prms.fHeight == 9))
        forwardAVX_K9(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);*/
    else
        forwardBASE(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate, weight, insz, input, outsz, output);
   
#else

    forwardBASE(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate, weight, insz, input, outsz, output);

#endif
}


void backwardGW_BASE(size_t kernel, size_t fWidth, size_t fHeight, size_t stride, size_t dilate,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){

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
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){
    
#ifdef SN_AVX
    
    if ((prms.fWidth == 3) && (prms.fHeight == 3) && false)
        backwardGW_AVX_K3(prms.kernel, prms.stride, prms.dilate,
            weight, insz, input, outsz, gradIn, gradOut, dWeightOut);
   /* else if ((prms.fWidth == 5) && (prms.fHeight == 5))
        forwardAVX_KRL5(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);
    else if ((prms.fWidth == 7) && (prms.fHeight == 7))
        forwardAVX_KRL7(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);
    else if ((prms.fWidth == 9) && (prms.fHeight == 9))
        forwardAVX_KRL9(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);*/
    else
        backwardGW_BASE(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate,
            weight, insz, input, outsz, gradIn, gradOut, dWeightOut);
    
#else

    backwardBASE(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate,
        weight, insz, input, outsz, gradIn, gradOut, dWeightOut);
#endif
}


void backwardG_Base(size_t kernel, size_t fWidth, size_t fHeight, size_t stride, size_t dilate,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut){

    size_t wStepByD = fWidth * fHeight,         // step weight by input         
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
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut){

#ifdef SN_AVX

    if ((prms.fWidth == 3) && (prms.fHeight == 3) && false)
        backwardG_AVX_K3(prms.kernel, prms.stride, prms.dilate,
        weight, insz, outsz, gradIn, gradOut);
    /* else if ((prms.fWidth == 5) && (prms.fHeight == 5))
    forwardAVX_KRL5(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);
    else if ((prms.fWidth == 7) && (prms.fHeight == 7))
    forwardAVX_KRL7(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);
    else if ((prms.fWidth == 9) && (prms.fHeight == 9))
    forwardAVX_KRL9(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);*/
    else
        backwardG_Base(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate,
        weight, insz, outsz, gradIn, gradOut);

#else

    backwardG_Base(prms.kernel, prms.stride, prms.dilate,
        weight, insz, outsz, gradIn, gradOut);
#endif
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