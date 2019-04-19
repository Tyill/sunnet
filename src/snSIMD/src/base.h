
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
#include <immintrin.h>

namespace SN_SIMD{

    const size_t LAYER_MAX_WIDTH = 800;
    const size_t LAYER_MAX_HEIGHT = 600;
    const size_t REG_CNT = 16;                   // registr count
    const size_t REG_BYTE_SZ = 32;               // registr byte size  (256 bit = 32 B = 8 float)
    const size_t L1_BYTE_SZ = 16 * 1024;         // L1 cache byte size (32 kB)
    const size_t L2_BYTE_SZ = 256 * 1024;        // L2 cache byte size (256 kB)
    const size_t L3_BYTE_SZ = 8 * 1024 * 1024;   // L3 cache byte size (2 MB/core)

#define LOAD_REG(in, reg)  __m256 reg = _mm256_loadu_ps(in);
#define LOAD_REG_FROM_MEM_3x3(in, reg) __m256 reg = _mm256_loadu_ps(in);                       

#define LOAD_1REG_FROM_MEM(m, in, reg) \
         LOAD_REG_FROM_MEM_ ## m ## x ## m ## (in, reg ## 0); in += m * m;

#define LOAD_2REG_FROM_MEM(m, in, reg) \
         LOAD_REG_FROM_MEM_ ## m ## x ## m ## (in, reg ## 0); in += m * m; \
         LOAD_REG_FROM_MEM_ ## m ## x ## m ## (in, reg ## 1); in += m * m; 

#define LOAD_3REG_FROM_MEM(m, in, reg) \
         LOAD_REG_FROM_MEM_ ## m ## x ## m ## (in, reg ## 0); in += m * m; \
         LOAD_REG_FROM_MEM_ ## m ## x ## m ## (in, reg ## 1); in += m * m; \
         LOAD_REG_FROM_MEM_ ## m ## x ## m ## (in, reg ## 2); in += m * m; 

#define LOAD_4REG_FROM_MEM(m, in, reg) \
         LOAD_REG_FROM_MEM_ ## m ## x ## m ## (in, reg ## 0); in += m * m; \
         LOAD_REG_FROM_MEM_ ## m ## x ## m ## (in, reg ## 1); in += m * m; \
         LOAD_REG_FROM_MEM_ ## m ## x ## m ## (in, reg ## 2); in += m * m; \
         LOAD_REG_FROM_MEM_ ## m ## x ## m ## (in, reg ## 3); in += m * m; 


#define SUMM_1REG(m, weight, arIn, arOut) \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, _mm256_loadu_ps(weight)), arOut); weight += (m) * (m);

#define SUMM_2REG(m, weight, arIn, arOut) \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, _mm256_loadu_ps(weight)), arOut); weight += (m) * (m); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 1, _mm256_loadu_ps(weight)), arOut); weight += (m) * (m);

#define SUMM_3REG(m, weight, arIn, arOut) \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, _mm256_loadu_ps(weight)), arOut); weight += (m) * (m); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 1, _mm256_loadu_ps(weight)), arOut); weight += (m) * (m); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 2, _mm256_loadu_ps(weight)), arOut); weight += (m) * (m);
         
#define SUMM_4REG(m, weight, arIn, arOut) \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, _mm256_loadu_ps(weight)), arOut); weight += (m) * (m); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 1, _mm256_loadu_ps(weight)), arOut); weight += (m) * (m); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 2, _mm256_loadu_ps(weight)), arOut); weight += (m) * (m); \
          arOut = _mm256_add_ps(_mm256_mul_ps(arIn ## 3, _mm256_loadu_ps(weight)), arOut); weight += (m) * (m);
    

    struct buf_t{

        SN_Base::snFloat* p = nullptr;
        SN_Base::snSize sz;

        buf_t(const SN_Base::snSize& size = SN_Base::snSize(0,0,0,0,0)) : sz(size) {
        
            if (size.size() > 0)
                p = (SN_Base::snFloat*)_mm_malloc(size.size() * sizeof(SN_Base::snFloat), 64);
        }

        ~buf_t() { 
            if (p) _mm_free(p);
        }
      
    };
    
    template<typename T>
    float horSummReg(T a);

    template<>
    float horSummReg<__m256>(__m256 a){

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

        if ((M == 3) && (D == 1)){

            for (size_t i = 0; i < outsz.h; ++i){
                 
                for (size_t j = 0; j < outsz.w; ++j){
                    
                    snFloat* pIn = input + S * insz.w * i + S * j;

                    for (size_t k = 0;  k < insz.d; ++k){

                        _mm_storeu_ps(output, _mm_loadu_ps(pIn));
                        _mm_storeu_ps(output + M, _mm_loadu_ps(pIn + insz.w));
                        _mm_storeu_ps(output + 2 * M, _mm_loadu_ps(pIn + 2 * insz.w));

                        pIn += insz.w * insz.h;
                        output += M * M;
                    }
                }
            }
        }
    };

    template<size_t M>
    SN_Base::snFloat getPeakOutput(size_t W, const SN_Base::snFloat* pIn, const SN_Base::snFloat* pW){

        SN_Base::snFloat ret = 0;

        for (size_t i = 0; i < W; i += 8){

            auto arIn = _mm256_set_ps(*(pIn +             (M * M - 1)), *(pIn +     M * M + (M * M - 1)), *(pIn + 2 * M * M + (M * M - 1)),
                                      *(pIn + 3 * M * M + (M * M - 1)), *(pIn + 4 * M * M + (M * M - 1)), *(pIn + 5 * M * M + (M * M - 1)),
                                      *(pIn + 6 * M * M + (M * M - 1)), *(pIn + 7 * M * M + (M * M - 1)));                    

            auto arW =  _mm256_set_ps(*(pW +             (M * M - 1)),  *(pW +     M * M + (M * M - 1)),  *(pW + 2 * M * M + (M * M - 1)),
                                      *(pW + 3 * M * M + (M * M - 1)),  *(pW + 4 * M * M + (M * M - 1)),  *(pW + 5 * M * M + (M * M - 1)),
                                      *(pW + 6 * M * M + (M * M - 1)),  *(pW + 7 * M * M + (M * M - 1)));                  

            ret += horSummReg(_mm256_mul_ps(arIn, arW));

            pIn += M * M * 8;
            pW += M * M * 8;
        }

        for (size_t i = 0; i < W % 8; ++i)
            ret += pIn[M * M * i + (M * M - 1)] * pW[M * M * i + (M * M - 1)];
        
        return ret;
    }       
};