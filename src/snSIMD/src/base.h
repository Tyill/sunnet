
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

    const size_t REG_CNT = 16;                   // registr count
    const size_t REG_BYTE_SZ = 32;               // registr byte size  (256 bit = 32 B = 8 float)
    const size_t L1_BYTE_SZ = 32 * 1024;         // L1 cache byte size (32 kB)
    const size_t L2_BYTE_SZ = 256 * 1024;        // L2 cache byte size (256 kB)
    const size_t L3_BYTE_SZ = 8 * 1024 * 1024;   // L3 cache byte size (2 MB/core)

#define CREATE_REG(reg) \
    __m256 reg = _mm256_setzero_ps();

#define CREATE_2REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps();

#define CREATE_3REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps(); \
    __m256 reg ## 3 = _mm256_setzero_ps();

#define CREATE_14REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps(); \
    __m256 reg ## 2 = _mm256_setzero_ps(); \
    __m256 reg ## 3 = _mm256_setzero_ps(); \
    __m256 reg ## 4 = _mm256_setzero_ps(); \
    __m256 reg ## 5 = _mm256_setzero_ps(); \
    __m256 reg ## 6 = _mm256_setzero_ps(); \
    __m256 reg ## 7 = _mm256_setzero_ps(); \
    __m256 reg ## 8 = _mm256_setzero_ps(); \
    __m256 reg ## 9 = _mm256_setzero_ps(); \
    __m256 reg ## 10 = _mm256_setzero_ps(); \
    __m256 reg ## 11 = _mm256_setzero_ps(); \
    __m256 reg ## 12 = _mm256_setzero_ps(); \
    __m256 reg ## 13 = _mm256_setzero_ps();

#define LOAD_REG(in, offs, reg) \
         reg = _mm256_loadu_ps(in + offs);

#define LOAD_3REG(in, offs, reg) \
         reg ## 0 = _mm256_loadu_ps(in + 0 * offs); \
         reg ## 1 = _mm256_loadu_ps(in + 1 * offs); \
         reg ## 2 = _mm256_loadu_ps(in + 2 * offs);

#define SUMM_REG(in, inOffs, arIn, arW, arO) \
        LOAD_REG(in, inOffs, arIn); arO = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO);
    
#define SUMM_14REG(in, inOffs, arIn, arW, arO) \
         LOAD_REG(in, 0 * inOffs, arIn); arO ## 0 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 0); \
         LOAD_REG(in, 1 * inOffs, arIn); arO ## 1 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 1); \
         LOAD_REG(in, 2 * inOffs, arIn); arO ## 2 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 2); \
         LOAD_REG(in, 3 * inOffs, arIn); arO ## 3 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 3); \
         LOAD_REG(in, 4 * inOffs, arIn); arO ## 4 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 4); \
         LOAD_REG(in, 5 * inOffs, arIn); arO ## 5 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 5); \
         LOAD_REG(in, 6 * inOffs, arIn); arO ## 6 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 6); \
         LOAD_REG(in, 7 * inOffs, arIn); arO ## 7 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 7); \
         LOAD_REG(in, 8 * inOffs, arIn); arO ## 8 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 8); \
         LOAD_REG(in, 9 * inOffs, arIn); arO ## 9 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 9); \
         LOAD_REG(in, 10 * inOffs, arIn); arO ## 10 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 10); \
         LOAD_REG(in, 11 * inOffs, arIn); arO ## 11 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 11); \
         LOAD_REG(in, 12 * inOffs, arIn); arO ## 12 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 12); \
         LOAD_REG(in, 13 * inOffs, arIn); arO ## 13 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 13);

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
       
        if (M == 1){

            for (size_t i = 0; i < outsz.h; ++i){

                for (size_t j = 0; j < outsz.w; ++j){
                    
                    snFloat* pIn = input + S * insz.w * i + S * j;

                    for (size_t k = 0; k < insz.d; ++k){

                        *pOut = *pIn;
                    
                        pIn += insz.w * insz.h;
                        pOut += M * M;
                    }
                }
            }
        }

        else if ((M == 3) && (D == 1)){

            for (size_t i = 0; i < outsz.h; ++i){
                 
                for (size_t j = 0; j < outsz.w; ++j){
                    
                    snFloat* pIn = input + S * insz.w * i + S * j;

                    for (size_t k = 0;  k < insz.d; ++k){

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));

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

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                        _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                        _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));

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

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                        _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                        _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));
                        _mm256_storeu_ps(pOut + 5 * M, _mm256_loadu_ps(pIn + 5 * insz.w));
                        _mm256_storeu_ps(pOut + 6 * M, _mm256_loadu_ps(pIn + 6 * insz.w));

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

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                        _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                        _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));
                        _mm256_storeu_ps(pOut + 5 * M, _mm256_loadu_ps(pIn + 5 * insz.w));
                        _mm256_storeu_ps(pOut + 6 * M, _mm256_loadu_ps(pIn + 6 * insz.w));
                        _mm256_storeu_ps(pOut + 7 * M, _mm256_loadu_ps(pIn + 7 * insz.w));
                        _mm256_storeu_ps(pOut + 8 * M, _mm256_loadu_ps(pIn + 8 * insz.w));

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

        ///////////////////////////////////////////////////
        

        else if ((M == 3) && (D == 2)){

            for (size_t i = 0; i < outsz.h; ++i){

                for (size_t j = 0; j < outsz.w; ++j){

                    snFloat* pIn = input + S * insz.w * i + S * j;

                    for (size_t k = 0; k < insz.d; ++k){
                                               
                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));

                        pIn += insz.w * insz.h;
                        pOut += M * M;
                    }
                }
            }
        }

        else if ((M == 5) && (D == 2)){

            for (size_t i = 0; i < outsz.h; ++i){

                for (size_t j = 0; j < outsz.w; ++j){

                    snFloat* pIn = input + S * insz.w * i + S * j;

                    for (size_t k = 0; k < insz.d; ++k){

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                        _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                        _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));

                        pIn += insz.w * insz.h;
                        pOut += M * M;
                    }
                }
            }
        }

        else if ((M == 7) && (D == 2)){

            for (size_t i = 0; i < outsz.h; ++i){

                for (size_t j = 0; j < outsz.w; ++j){

                    snFloat* pIn = input + S * insz.w * i + S * j;

                    for (size_t k = 0; k < insz.d; ++k){

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                        _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                        _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));
                        _mm256_storeu_ps(pOut + 5 * M, _mm256_loadu_ps(pIn + 5 * insz.w));
                        _mm256_storeu_ps(pOut + 6 * M, _mm256_loadu_ps(pIn + 6 * insz.w));

                        pIn += insz.w * insz.h;
                        pOut += M * M;
                    }
                }
            }
        }

        else if ((M == 9) && (D == 2)){

            for (size_t i = 0; i < outsz.h; ++i){

                for (size_t j = 0; j < outsz.w; ++j){

                    snFloat* pIn = input + S * insz.w * i + S * j;

                    for (size_t k = 0; k < insz.d; ++k){

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                        _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                        _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));
                        _mm256_storeu_ps(pOut + 5 * M, _mm256_loadu_ps(pIn + 5 * insz.w));
                        _mm256_storeu_ps(pOut + 6 * M, _mm256_loadu_ps(pIn + 6 * insz.w));
                        _mm256_storeu_ps(pOut + 7 * M, _mm256_loadu_ps(pIn + 7 * insz.w));
                        _mm256_storeu_ps(pOut + 8 * M, _mm256_loadu_ps(pIn + 8 * insz.w));

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


    };

    template<size_t M>
    SN_Base::snFloat getPeakOutput(size_t W, const SN_Base::snFloat* pIn, const SN_Base::snFloat* pW){

        SN_Base::snFloat ret = 0;
        
        for (size_t i = 0; i < W; ++i)
           ret += pIn[M * M * i + (M * M - 1)] * pW[M * M * i + (M * M - 1)];
    
        return ret;
    }       
};