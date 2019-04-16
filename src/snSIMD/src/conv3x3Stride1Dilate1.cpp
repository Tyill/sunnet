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


#include <omp.h>

#include "snBase/snBase.h"
#include "base.h"
#include "SimdEnable.h"

using namespace std;
using namespace SN_Base;

namespace SN_SIMD{
    
    void microCacheL1_M3x3_Stride1_Dilate1(snFloat* weight,
        const snSize& insz, snFloat* input, snFloat& output){

        // NCHW

        const size_t M = 3, W = insz.w, H = insz.h;

        snFloat IN_BUFF[M * M]{0}, OUT_BUFF[M * M]{output};

        LOAD_REG(OUT_BUFF, arO);

        __m256 arW = _mm256_setzero_ps();

        snFloat* pIn = input, *pW = weight;

        switch (insz.d){
        case 1: { LOAD_1REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_1REG(M, pW, ar, arW, arO); } break;
        case 2: { LOAD_2REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_2REG(M, pW, ar, arW, arO); } break;
        case 3: { 
            LOAD_3REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); 
                  
            arW = _mm256_loadu_ps(weight); 
            __m256 ars = _mm256_fmadd_ps(ar0, arW, arO);
            weight += M * M;
            
            arW = _mm256_loadu_ps(weight); arO = _mm256_fmadd_ps(ar1, arW, arO); weight += M * M; \
            arW = _mm256_loadu_ps(weight); arO = _mm256_fmadd_ps(ar2, arW, arO);
            
            //    SUMM_3REG(M, pW, ar, arW, arO); 
        
        } 
                break;
        case 4: { LOAD_4REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_4REG(M, pW, ar, arW, arO); } break;
        case 5: { LOAD_5REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_5REG(M, pW, ar, arW, arO); } break;
        case 6: { LOAD_6REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_6REG(M, pW, ar, arW, arO); } break;
        case 7: { LOAD_7REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_7REG(M, pW, ar, arW, arO); } break;
        case 8: { LOAD_8REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_8REG(M, pW, ar, arW, arO); } break;
        case 9: { LOAD_9REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_9REG(M, pW, ar, arW, arO); } break;
        case 10: { LOAD_10REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_10REG(M, pW, ar, arW, arO); } break;
        case 11: { LOAD_11REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_11REG(M, pW, ar, arW, arO); } break;
        case 12: { LOAD_12REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_12REG(M, pW, ar, arW, arO); } break;
        case 13: { LOAD_13REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_13REG(M, pW, ar, arW, arO); } break;
        default: { LOAD_14REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_14REG(M, pW, ar, arW, arO); } break;
        }

        output = horSummReg(arO);

        // add peak
        for (size_t i = 0; i < insz.d; ++i)
            output += (input + 2 * W + 2)[i * W * H] * weight[8 + i * M * M];
    }

    void macroCacheL2_M3x3_Stride1_Dilate1(snFloat* weight,
        const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

        // NCHW

        const size_t M = 3,        // mask
            S = 1,        // stride
            D = 1,        // dilate
            W = insz.w,
            H = insz.h,   // == M
            LAYER_CNT = min(size_t(REG_CNT - 2), min(insz.d, L1_BYTE_SZ / (sizeof(snFloat) * (W * M + M * M)))),
            IN_L1_SIZE = LAYER_CNT * W * M,
            W_L1_SIZE = LAYER_CNT * M * M;

        buf_t inL1Cache(IN_L1_SIZE), wL1Cache(W_L1_SIZE);

        snSize inCacheSz(insz.w, insz.h, LAYER_CNT);

        size_t inStep = insz.d / LAYER_CNT,
            inPeak = insz.d % LAYER_CNT;

        for (size_t k = 0; k < inStep; ++k){

            memcpy(inL1Cache.p, input + IN_L1_SIZE * k, IN_L1_SIZE * sizeof(snFloat));
            memcpy(wL1Cache.p, weight + W_L1_SIZE * k, W_L1_SIZE * sizeof(snFloat));

            for (size_t ox = 0; ox < outsz.w; ++ox){

                snFloat* pOut = output + ox;
                microCacheL1_M3x3_Stride1_Dilate1(wL1Cache.p, inCacheSz, inL1Cache.p + ox * S, *pOut);
            }
        }

        // count the remainder
        if (inPeak){

            inCacheSz.d = inPeak;

            memcpy(inL1Cache.p, input + IN_L1_SIZE * inStep, inPeak * W * M * sizeof(snFloat));
            memcpy(wL1Cache.p, weight + W_L1_SIZE * inStep, inPeak * M * M * sizeof(snFloat));

            for (size_t ox = 0; ox < outsz.w; ++ox){

                snFloat* pOut = output + ox;
                microCacheL1_M3x3_Stride1_Dilate1(wL1Cache.p, inCacheSz, inL1Cache.p + ox * S, *pOut);
            }
        }
    }

    void macroCacheL3_M3x3_Stride1_Dilate1(snFloat* weight,
        const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

        // NCHW

        const size_t M = 3,              // mask
            S = 1,              // stride
            D = 1,              // dilate
            W = insz.w,
            H = insz.h,
            LAYER_CNT = min(insz.d, L2_BYTE_SZ / (sizeof(snFloat) * (W * M * insz.d))),
            IN_L2_SIZE = LAYER_CNT * (W * M * insz.d);

        buf_t inL2Cache(IN_L2_SIZE);

        snSize inCacheSz(insz.w, M, LAYER_CNT);

        size_t inStep = insz.d / LAYER_CNT,
            inPeak = insz.d % LAYER_CNT;

        for (size_t k = 0; k < inStep; ++k){

            for (size_t oy = 0; oy < outsz.h; ++oy){

                for (size_t i = 0; i < LAYER_CNT; ++i)
                    memcpy(inL2Cache.p + W * M * i, input + oy * S * W + W * H * (i + k), W * M * sizeof(snFloat));

                snFloat* pOut = output + oy * outsz.w;
                macroCacheL2_M3x3_Stride1_Dilate1(weight, inCacheSz, inL2Cache.p, outsz, pOut);
            }
        }

        if (inPeak){

            inCacheSz.d = inPeak;

            for (size_t oy = 0; oy < outsz.h; ++oy){

                for (size_t i = 0; i < inPeak; ++i)
                    memcpy(inL2Cache.p + W * M * i, input + oy * S * W + W * H * (i + inStep), W * M * sizeof(snFloat));

                snFloat* pOut = output + oy * outsz.w;
                macroCacheL2_M3x3_Stride1_Dilate1(weight, inCacheSz, inL2Cache.p, outsz, pOut);
            }
        }
    }

    void convolution_M3x3_Stride1_Dilate1(snFloat* weight,
        const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

        // NCHW
        const size_t M = 3,     // mask
            S = 1,              // stride
            D = 1,              // dilate
            W = outsz.w,
            H = outsz.h,
            wStepByD = M * M,   // step weight by input
            wStepByK = wStepByD * insz.d,
            LAYER_CNT = min(outsz.d, L3_BYTE_SZ / (sizeof(snFloat) * (W * H))),
            OUT_L3_SIZE = LAYER_CNT * (W * H);

        buf_t outL3Cache(OUT_L3_SIZE);

        snSize outCacheSz(outsz.w, outsz.h, LAYER_CNT);

        size_t outStep = outsz.d / LAYER_CNT,
               outPeak = outsz.d % LAYER_CNT;

        for (size_t k = 0; k < outStep; ++k){

            memcpy(outL3Cache.p, output + OUT_L3_SIZE * k, OUT_L3_SIZE * sizeof(snFloat));

//#pragma omp parallel for
            for (int i = 0; i < int(LAYER_CNT); ++i){

                macroCacheL3_M3x3_Stride1_Dilate1(weight + wStepByK * i, insz, input, outCacheSz, outL3Cache.p + W * H * i);
            }
        }

        if (outPeak){

            memcpy(outL3Cache.p, output + OUT_L3_SIZE * outStep, outPeak * (W * H) * sizeof(snFloat));

            outCacheSz.d = outPeak;

//#pragma omp parallel for
            for (int i = 0; i < int(outPeak); ++i){

                macroCacheL3_M3x3_Stride1_Dilate1(weight + wStepByK * i, insz, input, outCacheSz, outL3Cache.p + W * H * i);
            }
        }
    }
}

