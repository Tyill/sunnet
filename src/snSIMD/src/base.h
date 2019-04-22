
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
#pragma once

#include "snBase/snBase.h"
#include <algorithm>
#include <iostream>
#include <immintrin.h>

namespace SN_SIMD{

    const size_t REG_CNT = 16;                   // registr count
    const size_t REG_BYTE_SZ = 32;               // registr byte size  (256 bit = 32 B = 8 float)
    const size_t L1_BYTE_SZ = 32 * 1024;         // L1 cache byte size (32 kB)
    const size_t L2_BYTE_SZ = 256 * 1024;        // L2 cache byte size (256 kB)
    const size_t L3_BYTE_SZ = 8 * 1024 * 1024;   // L3 cache byte size (2 MB/core)

#define LOAD_1REG_MEM1x1(in, reg) \
         __m256 reg ## 0 = _mm256_loadu_ps(in);

#define LOAD_1REG_MEM3x3(in, reg) \
         __m256 reg ## 0 = _mm256_loadu_ps(in);

#define LOAD_3REG_MEM5x5(in, reg) \
         __m256 reg ## 0 = _mm256_loadu_ps(in);     \
         __m256 reg ## 1 = _mm256_loadu_ps(in + 8); \
         __m256 reg ## 2 = _mm256_loadu_ps(in + 16); 

#define LOAD_6REG_MEM7x7(in, reg) \
         __m256 reg ## 0 = _mm256_loadu_ps(in);      \
         __m256 reg ## 1 = _mm256_loadu_ps(in + 8);  \
         __m256 reg ## 2 = _mm256_loadu_ps(in + 16); \
         __m256 reg ## 3 = _mm256_loadu_ps(in + 24); \
         __m256 reg ## 4 = _mm256_loadu_ps(in + 32); \
         __m256 reg ## 5 = _mm256_loadu_ps(in + 40); 

#define LOAD_10REG_MEM9x9(in, reg) \
         __m256 reg ## 0 = _mm256_loadu_ps(in);      \
         __m256 reg ## 1 = _mm256_loadu_ps(in + 8);  \
         __m256 reg ## 2 = _mm256_loadu_ps(in + 16); \
         __m256 reg ## 3 = _mm256_loadu_ps(in + 24); \
         __m256 reg ## 4 = _mm256_loadu_ps(in + 32); \
         __m256 reg ## 5 = _mm256_loadu_ps(in + 40); \
         __m256 reg ## 6 = _mm256_loadu_ps(in + 48); \
         __m256 reg ## 7 = _mm256_loadu_ps(in + 56); \
         __m256 reg ## 8 = _mm256_loadu_ps(in + 64); \
         __m256 reg ## 9 = _mm256_loadu_ps(in + 72); 

#define SUMM_1REG_MEM1x1(weight, arIn, arOut) \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, _mm256_loadu_ps(weight)), arOut);

#define SUMM_1REG_MEM3x3(weight, arIn, arOut) \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, _mm256_loadu_ps(weight)), arOut);
 
#define SUMM_3REG_MEM5x5(weight, arIn, arOut) \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, _mm256_loadu_ps(weight)), arOut);     \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 1, _mm256_loadu_ps(weight + 8)), arOut); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 2, _mm256_loadu_ps(weight + 16)), arOut);

#define SUMM_6REG_MEM7x7(weight, arIn, arOut) \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, _mm256_loadu_ps(weight)), arOut);      \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 1, _mm256_loadu_ps(weight + 8)), arOut);  \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 2, _mm256_loadu_ps(weight + 16)), arOut); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 3, _mm256_loadu_ps(weight + 24)), arOut); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 4, _mm256_loadu_ps(weight + 32)), arOut); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 5, _mm256_loadu_ps(weight + 40)), arOut);

#define SUMM_10REG_MEM9x9(weight, arIn, arOut) \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, _mm256_loadu_ps(weight)), arOut);      \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 1, _mm256_loadu_ps(weight + 8)), arOut);  \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 2, _mm256_loadu_ps(weight + 16)), arOut); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 3, _mm256_loadu_ps(weight + 24)), arOut); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 4, _mm256_loadu_ps(weight + 32)), arOut); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 5, _mm256_loadu_ps(weight + 40)), arOut); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 6, _mm256_loadu_ps(weight + 48)), arOut); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 7, _mm256_loadu_ps(weight + 56)), arOut); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 8, _mm256_loadu_ps(weight + 64)), arOut); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 9, _mm256_loadu_ps(weight + 72)), arOut);

    struct buf_t{

        SN_Base::snFloat* p = nullptr;
        SN_Base::snSize sz;

        buf_t(const SN_Base::snSize& size = SN_Base::snSize(0,0,0,0,0), size_t add = 0) : sz(size) {
        
            if (size.size() > 0)
                p = (SN_Base::snFloat*)_mm_malloc((size.size() + add) * sizeof(SN_Base::snFloat), 64);
        }

        ~buf_t() { 
            if (p) _mm_free(p);
        }
      
    };
    
    template<typename T>
    float horSummReg(T a);

    template<>
    static float horSummReg<__m256>(__m256 a){

        __m128 hi = _mm256_extractf128_ps(a, 1);
        __m128 lo = _mm256_extractf128_ps(a, 0);
        lo = _mm_add_ps(hi, lo);
        hi = _mm_movehl_ps(hi, lo);
        lo = _mm_add_ps(hi, lo);
        hi = _mm_shuffle_ps(lo, lo, 1);
        lo = _mm_add_ss(hi, lo);
        return _mm_cvtss_f32(lo);
    };

    template<size_t M, size_t S, size_t D>
    void reorderCHW2HCW(const SN_Base::snSize& insz, SN_Base::snFloat* input, const SN_Base::snSize& outsz, SN_Base::snFloat* output){

        SN_Base::snFloat* pOut = output;

        if ((M == 3) && (D == 1)){

            for (size_t i = 0; i < outsz.h; ++i){
                 
                for (size_t j = 0; j < outsz.w; ++j){
                    
                    snFloat* pIn = input + S * insz.w * i + S * j;

                    for (size_t k = 0;  k < insz.d; ++k){

                        _mm_storeu_ps(pOut, _mm_loadu_ps(pIn));
                        _mm_storeu_ps(pOut + M, _mm_loadu_ps(pIn + insz.w));
                        _mm_storeu_ps(pOut + 2 * M, _mm_loadu_ps(pIn + 2 * insz.w));

                        pIn += insz.w * insz.h;
                        pOut += M * M;
                    }
                }
            }
        }

        else if ((M == 5) && (D == 1)){

            for (size_t i = 0; i < outsz.h; ++i){

                for (size_t j = 0; j < outsz.w; ++j){

                    snFloat* pIn = input + S * insz.w * i + S * j;

                    for (size_t k = 0; k < insz.d; ++k){

                        _mm_storeu_ps(pOut, _mm_loadu_ps(pIn));
                        _mm_storeu_ps(pOut + M, _mm_loadu_ps(pIn + insz.w));
                        _mm_storeu_ps(pOut + 2 * M, _mm_loadu_ps(pIn + 2 * insz.w));
                        _mm_storeu_ps(pOut + 3 * M, _mm_loadu_ps(pIn + 3 * insz.w));
                        _mm_storeu_ps(pOut + 4 * M, _mm_loadu_ps(pIn + 4 * insz.w));

                        pIn += insz.w * insz.h;
                        pOut += M * M;
                    }
                }
            }
        }

        else if ((M == 7) && (D == 1)){
           
            for (size_t i = 0; i < outsz.h; ++i){

                for (size_t j = 0; j < outsz.w; ++j){

                    snFloat* pIn = input + S * insz.w * i + S * j;

                    for (size_t k = 0; k < insz.d; ++k){

                        _mm_storeu_ps(pOut, _mm_loadu_ps(pIn));
                        _mm_storeu_ps(pOut + M, _mm_loadu_ps(pIn + insz.w));
                        _mm_storeu_ps(pOut + 2 * M, _mm_loadu_ps(pIn + 2 * insz.w));
                        _mm_storeu_ps(pOut + 3 * M, _mm_loadu_ps(pIn + 3 * insz.w));
                        _mm_storeu_ps(pOut + 4 * M, _mm_loadu_ps(pIn + 4 * insz.w));
                        _mm_storeu_ps(pOut + 5 * M, _mm_loadu_ps(pIn + 5 * insz.w));
                        _mm_storeu_ps(pOut + 6 * M, _mm_loadu_ps(pIn + 6 * insz.w));

                        pIn += insz.w * insz.h;
                        pOut += M * M;
                    }
                }
            }
        }

        else if ((M == 9) && (D == 1)){

            for (size_t i = 0; i < outsz.h; ++i){

                for (size_t j = 0; j < outsz.w; ++j){

                    snFloat* pIn = input + S * insz.w * i + S * j;

                    for (size_t k = 0; k < insz.d; ++k){

                        _mm_storeu_ps(pOut, _mm_loadu_ps(pIn));
                        _mm_storeu_ps(pOut + M, _mm_loadu_ps(pIn + insz.w));
                        _mm_storeu_ps(pOut + 2 * M, _mm_loadu_ps(pIn + 2 * insz.w));
                        _mm_storeu_ps(pOut + 3 * M, _mm_loadu_ps(pIn + 3 * insz.w));
                        _mm_storeu_ps(pOut + 4 * M, _mm_loadu_ps(pIn + 4 * insz.w));
                        _mm_storeu_ps(pOut + 5 * M, _mm_loadu_ps(pIn + 5 * insz.w));
                        _mm_storeu_ps(pOut + 6 * M, _mm_loadu_ps(pIn + 6 * insz.w));
                        _mm_storeu_ps(pOut + 7 * M, _mm_loadu_ps(pIn + 7 * insz.w));
                        _mm_storeu_ps(pOut + 8 * M, _mm_loadu_ps(pIn + 8 * insz.w));

                        _mm_storeu_ps(pOut + 8, _mm_loadu_ps(pIn + 8));
                        _mm_storeu_ps(pOut + M + 8, _mm_loadu_ps(pIn + insz.w + 8));
                        _mm_storeu_ps(pOut + 2 * M + 8, _mm_loadu_ps(pIn + 2 * insz.w + 8));
                        _mm_storeu_ps(pOut + 3 * M + 8, _mm_loadu_ps(pIn + 3 * insz.w + 8));
                        _mm_storeu_ps(pOut + 4 * M + 8, _mm_loadu_ps(pIn + 4 * insz.w + 8));
                        _mm_storeu_ps(pOut + 5 * M + 8, _mm_loadu_ps(pIn + 5 * insz.w + 8));
                        _mm_storeu_ps(pOut + 6 * M + 8, _mm_loadu_ps(pIn + 6 * insz.w + 8));
                        _mm_storeu_ps(pOut + 7 * M + 8, _mm_loadu_ps(pIn + 7 * insz.w + 8));
                        _mm_storeu_ps(pOut + 8 * M + 8, _mm_loadu_ps(pIn + 8 * insz.w + 8));

                        pIn += insz.w * insz.h;
                        pOut += M * M;
                    }
                }
            }
        }

        // check border
        size_t bordw = (insz.w - M - (M - 1) * (D - 1)) % S;
        if (bordw != 0){
            
            for (size_t i = 0; i < outsz.h; ++i){
                
                pOut = output + (i * outsz.w + (outsz.w - 1)) * insz.d * M * M;

                for (size_t j = 0; j < insz.d; ++j){

                    for (size_t k = 0; k < M; ++k)
                        pOut[M * k + M - 1] = 0;
                    
                    pOut += M * M;
                }               
            }
        }

        size_t bordh = (insz.h - M - (M - 1) * (D - 1)) % S;
        if (bordh != 0){
                        
            for (size_t i = 0; i < outsz.w; ++i){
                   
                pOut = output + (outsz.w * (outsz.h - 1) + i) * insz.d * M * M;

                for (size_t j = 0; j < insz.d; ++j){

                    for (size_t k = 0; k < M; ++k)
                        pOut[M * (M - 1) + k] = 0;

                    pOut += M * M;
                }
            }
        }

    };

    template<size_t M>
    SN_Base::snFloat getPeakOutput(size_t W, const SN_Base::snFloat* pIn, const SN_Base::snFloat* pW){

        SN_Base::snFloat ret = 0;
                
        for (size_t i = 0; i < W; ++i)
            ret += pIn[M * M * i + (M * M - 1)] * pW[M * M * i + (M * M - 1)];
    
        return ret;
    }       
};