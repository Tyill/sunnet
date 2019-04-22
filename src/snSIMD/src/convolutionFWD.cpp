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
#include <thread>
#include "snBase/snBase.h"
#include "base.h"

using namespace std;
using namespace SN_Base;

namespace SN_SIMD{
     
        
    template<size_t M>
    class microL1
    {
        // Level 1: input calc

    public:
       
        microL1(const snSize& inHCWBuffSz){

            const size_t D = inHCWBuffSz.w / (M * M), // insz.d         
                         memSz = L1_BYTE_SZ / (2 * sizeof(snFloat)),  //   2 - < weights >
                         cacheLayerCnt = max(size_t(1), min(D, memSz / (M * M)));

            L1CacheSz = snSize(M * M, 1, cacheLayerCnt);
        };

        void operator()(snFloat* weight,
            const snSize& L1CacheSz_, snFloat* inHCWBuff, snFloat& output){

            const size_t cacheLayerCnt = L1CacheSz_.d;   // insz.d;

            __m256 arO = _mm256_setzero_ps();

            snFloat* pIn = inHCWBuff, *pW = weight;

            if (M > 1){
                for (size_t k = 0; k < cacheLayerCnt; ++k){

                    switch (M){
                    case 3: { LOAD_1REG_MEM3x3(pIn, ar); SUMM_1REG_MEM3x3(pW, ar, arO); } break;
                    case 5: { LOAD_3REG_MEM5x5(pIn, ar); SUMM_3REG_MEM5x5(pW, ar, arO); } break;
                    case 7: { LOAD_6REG_MEM7x7(pIn, ar); SUMM_6REG_MEM7x7(pW, ar, arO); } break;
                    case 9: { LOAD_10REG_MEM9x9(pIn, ar); SUMM_10REG_MEM9x9(pW, ar, arO); } break;
                    default: break;
                    }

                    pIn += M * M;
                    pW += M * M;
                }

                output += horSummReg<__m256>(arO);

                output += getPeakOutput<M>(cacheLayerCnt, inHCWBuff, weight);
            }
            else{

                for (size_t k = 0; k < cacheLayerCnt / 8; ++k){

                    LOAD_1REG_MEM1x1(pIn, ar); SUMM_1REG_MEM1x1(pW, ar, arO);
                    
                    pIn += 8;
                    pW += 8;
                }

                output += horSummReg<__m256>(arO);

                for (size_t k = 0; k < cacheLayerCnt % 8; ++k)
                    output += pIn[k] * pW[k];
            }          
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
                         memSz = L2_BYTE_SZ / sizeof(snFloat),
                         cacheLayerCnt = max(size_t(1), min(H, memSz / W));

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
                                 
            for (size_t k = 0; k < cacheStep; ++k){
                               
                refMicroL1_(pW, L1CacheSz, pIn, output);
                
                pW += L1Sz;
                pIn += L1Sz;
            }

            // count the remainder
            if (cachePeak){
                
                snSize cacheSz = L1CacheSz;
                cacheSz.d = cachePeak;
                            
                refMicroL1_(pW, cacheSz, pIn, output);
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
                         memSz = L3_BYTE_SZ / sizeof(snFloat),
                         cacheLayerCnt = max(size_t(1), min(inHCWBuffSz.d, memSz / (W * H)));
           
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

                    *(pOut + i) = bias;

                    refMacroL2_(pW, L2CacheSz, pIn + W * i, *(pOut + i));
                }

                pIn += L2Sz;
                pOut += cacheLayerCnt;
            }

            // count the remainder
            if (cachePeak){

                snSize cacheSz = L2CacheSz;
                cacheSz.d = cachePeak;
                                
                for (size_t i = 0; i < cachePeak; ++i){

                    *(pOut + i) = bias;

                    refMacroL2_(pW, cacheSz, pIn + W * i, *(pOut + i));
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
        buf_t inHCWBuff(snSize(M * M * insz.d, outsz.w, outsz.h), 8);
                
        reorderCHW2HCW<M, S, D>(insz, input, outsz, inHCWBuff.p);

        ///////////////////////////////////

        const size_t W = outsz.w,
                     H = outsz.h,
                     wStepByD = M * M,
                     wStepByK = wStepByD * insz.d,
                     wStepByN = wStepByK * outsz.d;
     
        macroL3<M> oMacroL3(inHCWBuff.sz);
       
        auto core = std::thread::hardware_concurrency();
        if (core == 0) core = 4;

#pragma omp parallel for num_threads(core)
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

    template <size_t M>
    void defaultFWD(size_t S, size_t D, snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

        const size_t wStepByD = M * M,          // step weight by input
            kernel = outsz.d,
            wStepByK = wStepByD * insz.d,       // step weight by output
            wStepByN = wStepByK * kernel,       // step weight by batch
            inStepByD = insz.w * insz.h,        // step in by input
            inStepByN = inStepByD * insz.d,     // step in by batch
            outStepByD = outsz.w * outsz.h,     // step out by input
            outStepByN = outStepByD * outsz.d;  // step out by batch

        size_t shareStepByN = kernel;           // for local mem
        snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

        auto core = std::thread::hardware_concurrency();
        if (core == 0) core = 4;

        // by batch
#pragma omp parallel for num_threads(core)
        for (int n = 0; n < int(insz.n); ++n){

            snFloat* outBuff = share + shareStepByN * n;
            snFloat In[wStepByD], W[wStepByD];

            for (size_t p = 0; p < outStepByD; ++p){

                size_t ox = p % outsz.w, oy = p / outsz.w,
                    posW = ox * S, posH = oy * S;

                memset(outBuff, 0, kernel * sizeof(snFloat));

                snFloat* pIn = input + inStepByN * n;
                snFloat* pW = weight;

                // on all in layers
                for (size_t d = 0; d < insz.d; ++d){

                    for (size_t c = 0; c < wStepByD; ++c){

                        size_t cx = c % M, cy = c / M;
                        In[c] = *(pIn + (cx + posW + cx * (D - 1)) + (cy + posH + cy * (D - 1)) * insz.w);
                    }

                    pW = weight + wStepByD * d;

                    // on all out layers
                    for (size_t k = 0; k < kernel; ++k){

                        for (size_t c = 0; c < wStepByD; ++c){

                            size_t cx = c % M, cy = c / M;
                            W[c] = *(pW + cx + cy * M);
                        }

                        __m256 arOut = _mm256_setzero_ps();

                        for (int z = 0; z < wStepByD / 8; ++z){

                            __m256 arIn = _mm256_loadu_ps(In + z * 8);

                            __m256 arW = _mm256_loadu_ps(W + z * 8);

                            arOut = _mm256_add_ps(arOut, _mm256_mul_ps(arIn, arW));
                        }

                        outBuff[k] += horSummReg<__m256>(arOut);

                        outBuff[k] += In[wStepByD - 1] * W[wStepByD - 1];

                        pW += wStepByK;
                    }

                    pIn += inStepByD;

                }

                snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
                pW = weight + wStepByN;

                // on all out layers
                for (size_t k = 0; k < kernel; ++k){

                    *pOut = outBuff[k] + *(pW + k); // + bias              

                    pOut += outStepByD;
                }
            }
        }

        free(share);
    }
    

    bool convolutionFWD(size_t M, size_t S, size_t D,
        snFloat* weight,
        const snSize& insz, snFloat* input,
        const snSize& outsz, snFloat* output){

      
        if ((insz.n > 1) || (S > 2) || (D > 2)){
  
#define dfwd(MS)   \
    if (M == MS){  \
        defaultFWD<MS>(S, D, weight, insz, input, outsz, output); return true; };

            dfwd(1)
            dfwd(3)
            dfwd(5)
            dfwd(7)
            dfwd(9)

            return false;
        }
#undef dfwd


        
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
  
            return false;
  
#undef cfwd

    };
};

