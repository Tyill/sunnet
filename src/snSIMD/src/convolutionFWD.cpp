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
        const snSize& inHCWBuffSz, snFloat* inHCWBuff, snFloat& output){

        // NCHW

        const size_t M = 3,
                     W = inHCWBuffSz.w / (M * M);   // insz.d;

        __m256 arO = _mm256_set1_ps(output);
                 
        size_t cacheStep = W / (REG_CNT - 12),
               cachePeak = W % (REG_CNT - 12);

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

        output = horSummReg(arO);
                     
        addPeakOutput<M>(W, inHCWBuff, weight, output);
    };
       
    template<size_t M>
    class microL1
    {
        // Level 1: calc 

    public:
        microL1(const snSize& inHCWBuffSz){

            const size_t W = inHCWBuffSz.w, // M * M * insz.d
                         H = inHCWBuffSz.h, // outsz.w
                         cacheLayerCnt = max(size_t(1), min(H, L1_BYTE_SZ / (sizeof(snFloat) * (W + W))));

            L1CacheSz = snSize(W, 1, cacheLayerCnt);
        };

        void operator()(snFloat* weight,
            const snSize& inHCWBuffSz, snFloat* inHCWBuff, snFloat& output){

            if (M == 3)
                microL1_M3x3(weight, inHCWBuffSz, inHCWBuff, output);
        };
            
        snSize L1CacheSz;
    };

    template<size_t M>
    class macroL2
    {
        // Level 2: input layers of width

    public:
        macroL2(const snSize& inHCWBuffSz) :
            refMicroL1_(inHCWBuffSz){

            const size_t W = inHCWBuffSz.w, // M * M * insz.d
                         H = inHCWBuffSz.h, // outsz.w
                         cacheLayerCnt = max(size_t(1), min(inHCWBuffSz.d, L2_BYTE_SZ / (sizeof(snFloat) * (W * H))));
           
            L2CacheSz = snSize(W, H, cacheLayerCnt);
        };

        void operator()(snFloat* weight, snFloat bias,
            const snSize& inHCWBuffSz, snFloat* inHCWBuff, const snSize& outsz, snFloat* output){

            // Down to level 1
                       
            const snSize& L1CacheSz = refMicroL1_.L1CacheSz;
           
            const size_t W = inHCWBuffSz.w,          // M * M * insz.d
                         H = inHCWBuffSz.h,          // outsz.w
                         cacheLayerCnt = L1CacheSz.d,// outsz.w
                         L1Sz = L1CacheSz.size(),
                         cacheStep = H / cacheLayerCnt,
                         cachePeak = H % cacheLayerCnt;

            snFloat* pIn = inHCWBuff,
                   * pW = weight,
                   * pOut = output;

            for (size_t k = 0; k < cacheStep; ++k){
                                                                
                for (size_t i = 0; i < cacheLayerCnt; ++i){                   
                   
                    refMicroL1_(pW, L1CacheSz, pIn + W * i, *(pOut + i));

                    *(pOut + i) += bias;
                }
                                
                pIn += L1Sz;
                pOut += cacheLayerCnt;
            }

            // count the remainder
            if (cachePeak){

                snSize cacheSz = L1CacheSz;
                cacheSz.d = cachePeak;

                pIn = inHCWBuff + L1Sz * cacheStep;
                pOut = output + cacheLayerCnt * cacheStep;

                for (size_t i = 0; i < cachePeak; ++i){
                                       
                    refMicroL1_(pW, cacheSz, pIn + W * i, *(pOut + i));

                    *(pOut + i) += bias;
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
        macroL3(const snSize& inHCWBuffSz, const snSize& outsz) :
            refMacroL2_(inHCWBuffSz){

            const size_t W = outsz.w, 
                         H = outsz.h, 
                         cacheLayerCnt = max(size_t(1), min(outsz.d, L3_BYTE_SZ / (sizeof(snFloat) * (W * H))));

            L3CacheSz = snSize(W, H, cacheLayerCnt);
        };

        void operator()(snFloat* weight, snFloat bias,
            const snSize& inHCWBuffSz, snFloat* inHCWBuff, const snSize& outsz, snFloat* output){

            // Down to level 2
                        
            const snSize& L2CacheSz = refMacroL2_.L2CacheSz;
            
            const size_t W = L2CacheSz.w,             // M * M * insz.d
                         H = L2CacheSz.h,             // outsz.w
                         cacheLayerCnt = L2CacheSz.d, // outsz.h
                         L2Sz = L2CacheSz.size(),
                         cacheStep = inHCWBuffSz.d / cacheLayerCnt,
                         cachePeak = inHCWBuffSz.d % cacheLayerCnt;
                
            snFloat* pIn = inHCWBuff,
                   * pOut = output;
                        
            for (size_t k = 0; k < cacheStep; ++k){
                  
                for (size_t i = 0; i < cacheLayerCnt; ++i){
                                        
                    refMacroL2_(weight, bias, L2CacheSz, pIn + W * H * i, outsz, pOut + outsz.w * i);
                }
                pIn += L2Sz;
                pOut += outsz.w * cacheLayerCnt;
            }

            if (cachePeak){
                                       
                snSize cacheSz = L2CacheSz;
                cacheSz.d = cachePeak;

                pIn = inHCWBuff + L2Sz * cacheStep;
                pOut = output + outsz.w * cacheLayerCnt * cacheStep;

                for (size_t i = 0; i < cachePeak; ++i){
                   
                    refMacroL2_(weight, bias, cacheSz, pIn, outsz, pOut);
                
                    pIn += W * H;
                    pOut += outsz.w;
                }
            }
        };

        snSize L3CacheSz;

    private:
        macroL2<M> refMacroL2_;
       
    };
            
    template<size_t M, size_t S, size_t D>
    void convolutionFWD(snFloat* weight,
        const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

        for (int n = 0; n < int(insz.n); ++n){
        
            /// Reorder input
            buf_t inHCWBuff(snSize(M * M * insz.d, outsz.w, outsz.h));

            snFloat* pIn = input + insz.w * insz.h * insz.d * n;

            reorderCHW2HCW<M, S, D>(insz, pIn, outsz, inHCWBuff.p);

            ///////////////////////////////////
                    
            macroL3<M> oMacroL3(inHCWBuff.sz, outsz);

            const snSize& L3CacheSz = oMacroL3.L3CacheSz;

            const size_t W = outsz.w,
                         H = outsz.h,
                         wStepByD = M * M,
                         wStepByK = wStepByD * insz.d,
                         wStepByN = wStepByK * outsz.d,
                         cacheLayerCnt = L3CacheSz.d,
                         L3Sz = L3CacheSz.size(),
                         cacheStep = outsz.d / cacheLayerCnt,
                         cachePeak = outsz.d % cacheLayerCnt;

            ///////////////////////////////////

            snFloat* pOut = output + outsz.w * outsz.h * outsz.d * n;
            
            for (size_t k = 0; k < cacheStep; ++k){
                
#pragma omp parallel for num_threads(4)
                for (int i = 0; i < int(cacheLayerCnt); ++i){

                    oMacroL3(weight + wStepByK * (i + k * cacheLayerCnt), *(weight + wStepByN + i),
                        inHCWBuff.sz, inHCWBuff.p, L3CacheSz, pOut + W * H * i);                             
                }

                pOut += L3Sz;
            }

            if (cachePeak){
                               
                snSize cacheSz = L3CacheSz;
                cacheSz.d = cachePeak;

                pOut = output + outsz.w * outsz.h * outsz.d * n + L3Sz * cacheStep;

#pragma omp parallel for num_threads(4)
                for (int i = 0; i < int(cachePeak); ++i){

                    oMacroL3(weight + wStepByK * (i + cacheStep * cacheLayerCnt), *(weight + wStepByN + i),
                        inHCWBuff.sz, inHCWBuff.p, cacheSz, pOut + W * H * i);
                }
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
    };
};

