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

using namespace std;
using namespace SN_Base;

namespace SN_SIMD{
    
  
    template<size_t D>
    void microL1_M3x3(snFloat* weight,
        const snSize& insz, snFloat* input, snFloat& output){

        // NCHW

        const size_t M = 3, W = insz.w, H = insz.h;

        snFloat IN_BUFF[M * M]{0}, OUT_BUFF[M * M]{output};

        LOAD_REG(OUT_BUFF, arO);

        __m256 arW = _mm256_setzero_ps();
                
        size_t inStep = insz.d / (REG_CNT - 2),
               inPeak = insz.d % (REG_CNT - 2);

        snFloat* pIn = input, *pW = weight;

        for (size_t k = 0; k < inStep; ++k){

            LOAD_14REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_14REG(M, pW, ar, arW, arO);
        }
                
        switch (inPeak){
           case 0: break;
           case 1: { LOAD_1REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_1REG(M, pW, ar, arW, arO); } break;
           case 2: { LOAD_2REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_2REG(M, pW, ar, arW, arO); } break;
           case 3: { LOAD_3REG_FROM_BUFF(3, 1, pIn, IN_BUFF, W, H, ar); SUMM_3REG(M, pW, ar, arW, arO); } break;
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
           default: break;
        }

        output = horSummReg(arO);

        // add peak
        for (size_t i = 0; i < insz.d; ++i)
            output += (input + 2 * W + 2)[i * W * H] * weight[8 + i * M * M];
    };

    template<size_t M, size_t D>
    class microL1
    {
    public:
        microL1(const snSize& insz){

            const size_t W = insz.w,
                         LAYER_CNT = min(insz.d, L1_BYTE_SZ / (sizeof(snFloat) * (W * M + M * M)));

            inBuff.resize(snSize(W, M, LAYER_CNT));
            wBuff.resize(snSize(M, M, LAYER_CNT));
        };

        void operator()(snFloat* weight,
            const snSize& insz, snFloat* input, snFloat& output){

            if (M == 3)
                microL1_M3x3<D>(weight, insz, input, output);
        };
            
        buf_t inBuff, wBuff;
    };

    template<size_t M, size_t S, size_t D>
    class macroL2
    {
    public:
        macroL2(const snSize& insz) :
            refMicroL1_(insz){

            const size_t W = insz.w,
                         LAYER_CNT = min(insz.d, L2_BYTE_SZ / (sizeof(snFloat) * (W * M)));

            inBuff.resize(snSize(W, M, LAYER_CNT));
        };

        void operator()(snFloat* weight,
            const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

            // NCHW
                       
            snSize& inCacheSz = refMicroL1_.inBuff.sz;
            snSize& wCacheSz = refMicroL1_.wBuff.sz;
            snFloat* inL1Cache = refMicroL1_.inBuff.p;
            snFloat* wL1Cache = refMicroL1_.wBuff.p;

            const size_t W = insz.w,
                         LAYER_CNT = inCacheSz.d,
                         IN_L1_SIZE = inCacheSz.size(),
                         W_L1_SIZE = wCacheSz.size();

            size_t inStep = insz.d / LAYER_CNT,
                   inPeak = insz.d % LAYER_CNT;

            for (size_t k = 0; k < inStep; ++k){

                memcpy(inL1Cache, input + IN_L1_SIZE * k, IN_L1_SIZE * sizeof(snFloat));
                memcpy(wL1Cache, weight + W_L1_SIZE * k, W_L1_SIZE * sizeof(snFloat));

                for (size_t ox = 0; ox < outsz.w; ++ox){

                    snFloat* pOut = output + ox;
                    refMicroL1_(wL1Cache, inCacheSz, inL1Cache + ox * S, *pOut);
                }
            }

            // count the remainder
            if (inPeak){

                snSize cacheSz = inCacheSz;
                cacheSz.d = inPeak;

                memcpy(inL1Cache, input + IN_L1_SIZE * inStep, inPeak * W * M * sizeof(snFloat));
                memcpy(wL1Cache, weight + W_L1_SIZE * inStep, inPeak * M * M * sizeof(snFloat));

                for (size_t ox = 0; ox < outsz.w; ++ox){

                    snFloat* pOut = output + ox;
                    refMicroL1_(wL1Cache, cacheSz, inL1Cache + ox * S, *pOut);
                }
            }
        };

        buf_t inBuff;

    private:
        microL1<M, D> refMicroL1_;
       
    };

    template<size_t M, size_t S, size_t D>
    class macroL3
    {
    public:
        macroL3(const snSize& insz, const snSize& outsz) :
            refMacroL2_(insz){

            const size_t W = outsz.w,
                         H = outsz.h,
                         LAYER_CNT = min(outsz.d, L3_BYTE_SZ / (sizeof(snFloat) * (W * H)));

            outBuff.resize(snSize(W, H, LAYER_CNT));
        };

        void operator()(snFloat* weight,
            const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

            // NCHW
                        
            snSize& inCacheSz = refMacroL2_.inBuff.sz;
            snFloat* inL2Cache = refMacroL2_.inBuff.p;

            const size_t W = insz.w,
                         H = insz.h,
                         LAYER_CNT = inCacheSz.d;

            size_t inStep = insz.d / LAYER_CNT,
                   inPeak = insz.d % LAYER_CNT;
                       
            for (size_t k = 0; k < inStep; ++k){

                for (size_t oy = 0; oy < outsz.h; ++oy){

                    for (size_t i = 0; i < LAYER_CNT; ++i)
                        memcpy(inL2Cache + W * M * i, input + oy * S * W + W * H * (i + k * LAYER_CNT), W * M * sizeof(snFloat));

                    snFloat* pOut = output + oy * outsz.w;
                    refMacroL2_(weight, inCacheSz, inL2Cache, outsz, pOut);
                }
            }

            if (inPeak){
                               
                snSize cacheSz = inCacheSz;
                cacheSz.d = inPeak;

                for (size_t oy = 0; oy < outsz.h; ++oy){

                    for (size_t i = 0; i < inPeak; ++i)
                        memcpy(inL2Cache + W * M * i, input + oy * S * W + W * H * (i + inStep * LAYER_CNT), W * M * sizeof(snFloat));

                    snFloat* pOut = output + oy * outsz.w;
                    refMacroL2_(weight, cacheSz, inL2Cache, outsz, pOut);
                }
            }
        };

        buf_t outBuff;

    private:
        macroL2<M, S, D> refMacroL2_;
       
    };
            
    template<size_t M, size_t S, size_t D>
    void convolutionFWD(snFloat* weight,
        const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

        // NCHW
      
        macroL3<M, S, D> oMacroL3(insz, outsz);
               
        snSize& outCacheSz = oMacroL3.outBuff.sz;
        snFloat* outL3Cache = oMacroL3.outBuff.p;

        const size_t W = outsz.w,
                     H = outsz.h,
                     wStepByD = M * M,   // step weight by input
                     wStepByK = wStepByD * insz.d,
                     LAYER_CNT = outCacheSz.d,
                     OUT_L3_SIZE = outCacheSz.size();

        size_t outStep = outsz.d / LAYER_CNT,
               outPeak = outsz.d % LAYER_CNT;

//#pragma omp parallel for
        for (int n = 0; n < int(insz.n); ++n){
                       
            snFloat* pIn = input + insz.w * insz.h * insz.d * n,
                   * pOut = output + outsz.w * outsz.h * outsz.d * n;
            
            for (size_t k = 0; k < outStep; ++k){

                memcpy(outL3Cache, pOut + OUT_L3_SIZE * k, OUT_L3_SIZE * sizeof(snFloat));

                for (size_t i = 0; i < LAYER_CNT; ++i){

                    oMacroL3(weight + wStepByK * (i + k * LAYER_CNT),
                        insz, pIn, outCacheSz, outL3Cache + W * H * i);
                }

                memcpy(pOut + OUT_L3_SIZE * k, outL3Cache, OUT_L3_SIZE * sizeof(snFloat));
            }

            if (outPeak){

                memcpy(outL3Cache, pOut + OUT_L3_SIZE * outStep, outPeak * (W * H) * sizeof(snFloat));

                snSize cacheSz = outCacheSz;
                cacheSz.d = outPeak;

                for (size_t i = 0; i < outPeak; ++i){

                    oMacroL3(weight + wStepByK * (i + outStep * LAYER_CNT),
                        insz, pIn, cacheSz, outL3Cache + W * H * i);
                }

                memcpy(pOut + OUT_L3_SIZE * outStep, outL3Cache, outPeak * (W * H) * sizeof(snFloat));
            }
        }
    }

    bool convolutionFWD(size_t M, size_t S, size_t D,
        snFloat* weight,
        const snSize& insz, snFloat* input,
        const snSize& outsz, snFloat* output){

        if ((insz.w > LAYER_MAX_WIDTH) || (insz.h > LAYER_MAX_HEIGHT))
            return false;

#define cfwd(MS, SS, DS)                       \
    if ((M == MS) && (S == SS) && (D == DS)){  \
        convolutionFWD<MS, SS, DS>(weight, insz, input, outsz, output); return true; };

            cfwd(1, 1, 1)
            cfwd(3, 1, 1)
            cfwd(5, 1, 1)
            cfwd(7, 1, 1)
            cfwd(9, 1, 1)

            cfwd(1, 2, 1)
            cfwd(3, 2, 1)
            cfwd(5, 2, 1)
            cfwd(7, 2, 1)
            cfwd(9, 2, 1)

            cfwd(1, 1, 2)
            cfwd(3, 1, 2)
            cfwd(5, 1, 2)
            cfwd(7, 1, 2)
            cfwd(9, 1, 2)

            cfwd(1, 2, 2)
            cfwd(3, 2, 2)
            cfwd(5, 2, 2)
            cfwd(7, 2, 2)
            cfwd(9, 2, 2)

#undef cfwd

            return false;
    }
}

