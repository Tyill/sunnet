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
#include <iostream>

#include "snBase/snBase.h"
#include "base.h"

using namespace std;
using namespace SN_Base;

namespace SN_SIMD{
      
    void microL1_M3x3(snFloat* weight,
        const snSize& L1CacheSz_, snFloat* inHCWBuff, snFloat& output){

        // NCHW

        const size_t M = 3,
                     cacheLayerCnt = L1CacheSz_.d;   // insz.d;

        __m256 arO = _mm256_set1_ps(output);
                 
        size_t cacheStep = cacheLayerCnt / (REG_CNT - 12),
               cachePeak = cacheLayerCnt % (REG_CNT - 12);

        snFloat* pIn = inHCWBuff, *pW = weight;

        for (size_t k = 0; k < cacheStep; ++k){

            LOAD_4REG_FROM_MEM(3, pIn, ar); SUMM_4REG(M, pW, ar, arO);
        }
                
        switch (cachePeak){
           case 0: break;
           case 1: { LOAD_1REG_FROM_MEM(3, pIn, ar); SUMM_1REG(M, pW, ar, arO); } break;
           case 2: { LOAD_2REG_FROM_MEM(3, pIn, ar); SUMM_2REG(M, pW, ar, arO); } break;
           case 3: { LOAD_3REG_FROM_MEM(3, pIn, ar); SUMM_3REG(M, pW, ar, arO); } break;
           default: break;
        }

        output += horSummReg(arO);
                     
        output += getPeakOutput<M>(cacheLayerCnt, inHCWBuff, weight);
    };
       
    template<size_t M>
    class microL1
    {
        // Level 1: input calc

    public:
        microL1(const snSize& inHCWBuffSz){

            const size_t D = inHCWBuffSz.w / (M * M), // insz.d              2 -< weights >       
                         cacheLayerCnt = max(size_t(1), min(D, L1_BYTE_SZ / (M * M * sizeof(snFloat))));

            L1CacheSz = snSize(M * M, 1, cacheLayerCnt);
        };

        void operator()(snFloat* weight,
            const snSize& L1CacheSz_, snFloat* inHCWBuff, snFloat& output){

            if (M == 3)
                microL1_M3x3(weight, L1CacheSz_, inHCWBuff, output);
        };

        snSize L1CacheSz;
    };

    template<size_t M>
    class macroL2
    {
        // Level 2: input by depth 

    public:
        macroL2(const snSize& inHCWBuffSz) :
            refMicroL1_(inHCWBuffSz){

            const size_t W = inHCWBuffSz.w, // M * M * insz.d
                         H = inHCWBuffSz.h, // outsz.w
                         cacheLayerCnt = max(size_t(1), min(H, L2_BYTE_SZ / (W * sizeof(snFloat))));

            L2CacheSz = snSize(M * M, inHCWBuffSz.w / (M * M), cacheLayerCnt);
        };

        void operator()(snFloat* weight,
            const snSize& L2CacheSz_, snFloat* inHCWBuff, snFloat& output){

            // Down to level 1

            const snSize& L1CacheSz = refMicroL1_.L1CacheSz;

            const size_t H = L2CacheSz_.h,           // insz.d
                         cacheLayerCnt = L1CacheSz.d,// insz.d
                         L1Sz = L1CacheSz.size(),
                         cacheStep = H / cacheLayerCnt,
                         cachePeak = H % cacheLayerCnt;

            snFloat* pIn = inHCWBuff,
                   * pW = weight;

            // for only input 
            for (size_t k = 0; k < cacheStep; ++k){
                               
                refMicroL1_(pW, L1CacheSz, pIn, output);
                
                pW += L1Sz;
                pIn += L1Sz;
            }

            // count the remainder
            if (cachePeak){

                snSize cacheSz = L1CacheSz;
                cacheSz.d = cachePeak;
                   
                const size_t L1CSz = cacheSz.size();

                for (size_t i = 0; i < cachePeak; ++i){

                    refMicroL1_(pW, cacheSz, pIn, output);

                    pW += L1CSz;
                    pIn += L1CSz;
                }
            }
        };
            
        snSize L2CacheSz;

    private:
        microL1<M> refMicroL1_;
    };

    template<size_t M>
    class macroL3
    {
        // Level 3: input layers of width * height

    public:
        macroL3(const snSize& inHCWBuffSz) :
            refMacroL2_(inHCWBuffSz){

            const size_t W = inHCWBuffSz.w, // M * M * insz.d
                         H = inHCWBuffSz.h, // outsz.w
                         cacheLayerCnt = max(size_t(1), min(inHCWBuffSz.d, L3_BYTE_SZ / (W * H * sizeof(snFloat))));
           
            L3CacheSz = snSize(W, H, cacheLayerCnt);
        };

        void operator()(snFloat* weight, snFloat bias,
            const snSize& L3CacheSz_, snFloat* inHCWBuff, const snSize& outsz, snFloat* output){

            // Down to level 2

            const snSize& L2CacheSz = refMacroL2_.L2CacheSz;

            const size_t W = L3CacheSz_.w,          // M * M * insz.d
                         H = L3CacheSz_.h,          // outsz.w
                         cacheLayerCnt = L2CacheSz.d,// outsz.w
                         L2Sz = L2CacheSz.size(),
                         cacheStep = H / cacheLayerCnt,
                         cachePeak = H % cacheLayerCnt;

            snFloat* pIn = inHCWBuff,
                   * pW = weight,
                   * pOut = output;

            for (size_t k = 0; k < cacheStep; ++k){

                for (size_t i = 0; i < cacheLayerCnt; ++i){

                    refMacroL2_(pW, L2CacheSz, pIn + W * i, *(pOut + i));

                    *(pOut + i) += bias;
                }

                pIn += L2Sz;
                pOut += cacheLayerCnt;
            }

            // count the remainder
            if (cachePeak){

                snSize cacheSz = L2CacheSz;
                cacheSz.d = cachePeak;
                                
                for (size_t i = 0; i < cachePeak; ++i){

                    refMacroL2_(pW, cacheSz, pIn + W * i, *(pOut + i));

                    *(pOut + i) += bias;
                }
            }
        };

        snSize L3CacheSz;

    private:
        macroL2<M> refMacroL2_;
       
    };
         
    template<size_t M>
    void macroCommon(macroL3<M>& refMacroL3, snFloat* weight, snFloat bias,
        const snSize& inHCWBuffSz, snFloat* inHCWBuff, const snSize& outsz, snFloat* output){
            
        // Down to level 3

        const snSize& L3CacheSz = refMacroL3.L3CacheSz;

        const size_t W = L3CacheSz.w,             // M * M * insz.d
                     H = L3CacheSz.h,             // outsz.w
                     cacheLayerCnt = L3CacheSz.d, // outsz.h
                     L3Sz = L3CacheSz.size(),
                     cacheStep = inHCWBuffSz.d / cacheLayerCnt,
                     cachePeak = inHCWBuffSz.d % cacheLayerCnt;

        snFloat* pIn = inHCWBuff,
               * pOut = output;

        for (size_t k = 0; k < cacheStep; ++k){

            for (size_t i = 0; i < cacheLayerCnt; ++i){

                refMacroL3(weight, bias, L3CacheSz, pIn + W * H * i, outsz, pOut + outsz.w * i);
            }
            pIn += L3Sz;
            pOut += outsz.w * cacheLayerCnt;
        }

        if (cachePeak){

            snSize cacheSz = L3CacheSz;
            cacheSz.d = cachePeak;

            for (size_t i = 0; i < cachePeak; ++i){

                refMacroL3(weight, bias, cacheSz, pIn, outsz, pOut);

                pIn += W * H;
                pOut += outsz.w;
            }
        }
    }

    template<size_t M, size_t S, size_t D>
    void convolutionFWD(snFloat* weight,
        const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){
                
        /// Reorder input
        buf_t inHCWBuff(snSize(M * M * insz.d, outsz.w, outsz.h));
                
        reorderCHW2HCW<M, S, D>(insz, input, outsz, inHCWBuff.p);

        ///////////////////////////////////

        const size_t W = outsz.w,
                     H = outsz.h,
                     wStepByD = M * M,
                     wStepByK = wStepByD * insz.d,
                     wStepByN = wStepByK * outsz.d;
                 
        macroL3<M> oMacroL3(inHCWBuff.sz);

#pragma omp parallel for num_threads(4)
        for (int i = 0; i < int(outsz.d); ++i){

            macroCommon(oMacroL3, 
                        weight + wStepByK * i,
                        *(weight + wStepByN + i),
                        inHCWBuff.sz,
                        inHCWBuff.p,
                        outsz,
                        output + W * H * i);
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
    };
};

