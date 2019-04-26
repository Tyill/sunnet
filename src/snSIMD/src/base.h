
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
    const size_t REG_BYTE_SZ = 32;               // registr byte size  (256bit = 32B = 8float)
    const size_t L1_BYTE_SZ = 32 * 1024;         // L1 cache byte size (32kB / core)
    const size_t L2_BYTE_SZ = 256 * 1024;        // L2 cache byte size (256kB / core)
    const size_t L3_BYTE_SZ = 8 * 1024 * 1024;   // L3 cache byte size (8MB / all core)

#define CREATE_REG(reg) \
    __m256 reg = _mm256_setzero_ps();

#define CREATE_2REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps();

#define CREATE_3REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps(); \
    __m256 reg ## 2 = _mm256_setzero_ps();

#define CREATE_4REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps(); \
    __m256 reg ## 2 = _mm256_setzero_ps(); \
    __m256 reg ## 3 = _mm256_setzero_ps();

#define CREATE_5REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps(); \
    __m256 reg ## 2 = _mm256_setzero_ps(); \
    __m256 reg ## 3 = _mm256_setzero_ps(); \
    __m256 reg ## 4 = _mm256_setzero_ps();

#define CREATE_6REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps(); \
    __m256 reg ## 2 = _mm256_setzero_ps(); \
    __m256 reg ## 3 = _mm256_setzero_ps(); \
    __m256 reg ## 4 = _mm256_setzero_ps(); \
    __m256 reg ## 5 = _mm256_setzero_ps();

#define CREATE_7REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps(); \
    __m256 reg ## 2 = _mm256_setzero_ps(); \
    __m256 reg ## 3 = _mm256_setzero_ps(); \
    __m256 reg ## 4 = _mm256_setzero_ps(); \
    __m256 reg ## 5 = _mm256_setzero_ps(); \
    __m256 reg ## 6 = _mm256_setzero_ps();

#define CREATE_8REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps(); \
    __m256 reg ## 2 = _mm256_setzero_ps(); \
    __m256 reg ## 3 = _mm256_setzero_ps(); \
    __m256 reg ## 4 = _mm256_setzero_ps(); \
    __m256 reg ## 5 = _mm256_setzero_ps(); \
    __m256 reg ## 6 = _mm256_setzero_ps(); \
    __m256 reg ## 7 = _mm256_setzero_ps();

#define CREATE_9REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps(); \
    __m256 reg ## 2 = _mm256_setzero_ps(); \
    __m256 reg ## 3 = _mm256_setzero_ps(); \
    __m256 reg ## 4 = _mm256_setzero_ps(); \
    __m256 reg ## 5 = _mm256_setzero_ps(); \
    __m256 reg ## 6 = _mm256_setzero_ps(); \
    __m256 reg ## 7 = _mm256_setzero_ps(); \
    __m256 reg ## 8 = _mm256_setzero_ps();

#define CREATE_10REG(reg) \
    __m256 reg ## 0 = _mm256_setzero_ps(); \
    __m256 reg ## 1 = _mm256_setzero_ps(); \
    __m256 reg ## 2 = _mm256_setzero_ps(); \
    __m256 reg ## 3 = _mm256_setzero_ps(); \
    __m256 reg ## 4 = _mm256_setzero_ps(); \
    __m256 reg ## 5 = _mm256_setzero_ps(); \
    __m256 reg ## 6 = _mm256_setzero_ps(); \
    __m256 reg ## 7 = _mm256_setzero_ps(); \
    __m256 reg ## 8 = _mm256_setzero_ps(); \
    __m256 reg ## 9 = _mm256_setzero_ps();

#define CREATE_11REG(reg) \
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
    __m256 reg ## 10 = _mm256_setzero_ps();

#define CREATE_12REG(reg) \
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
    __m256 reg ## 11 = _mm256_setzero_ps();

#define CREATE_13REG(reg) \
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
    __m256 reg ## 12 = _mm256_setzero_ps();

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

#define LOAD_5REG(in, offs, reg) \
         reg ## 0 = _mm256_loadu_ps(in + 0 * offs); \
         reg ## 1 = _mm256_loadu_ps(in + 1 * offs); \
         reg ## 2 = _mm256_loadu_ps(in + 2 * offs); \
         reg ## 3 = _mm256_loadu_ps(in + 3 * offs); \
         reg ## 4 = _mm256_loadu_ps(in + 4 * offs);

#define LOAD_6REG(in, offs, reg) \
         reg ## 0 = _mm256_loadu_ps(in + 0 * offs); \
         reg ## 1 = _mm256_loadu_ps(in + 1 * offs); \
         reg ## 2 = _mm256_loadu_ps(in + 2 * offs); \
         reg ## 3 = _mm256_loadu_ps(in + 3 * offs); \
         reg ## 4 = _mm256_loadu_ps(in + 4 * offs); \
         reg ## 5 = _mm256_loadu_ps(in + 5 * offs);

#define SUMM_3x3REG_1OUT(arIn, arW, arO) \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, arW ## 0), arO); \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 1, arW ## 1), arO); \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 2, arW ## 2), arO);

#define SUMM_3x3REG_2OUT(pIn, inOffs, arIn, arW, arO) \
         LOAD_3REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_3REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 1);

#define SUMM_3x3REG_3OUT(pIn, inOffs,arIn, arW, arO) \
         LOAD_3REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_3REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 1); \
         LOAD_3REG(pIn + 2 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 2);

#define SUMM_3x3REG_4OUT(pIn, inOffs,arIn, arW, arO) \
         LOAD_3REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_3REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 1); \
         LOAD_3REG(pIn + 2 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 2); \
         LOAD_3REG(pIn + 3 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 3);

#define SUMM_3x3REG_5OUT(pIn, inOffs,arIn, arW, arO) \
         LOAD_3REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_3REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 1); \
         LOAD_3REG(pIn + 2 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 2); \
         LOAD_3REG(pIn + 3 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 3); \
         LOAD_3REG(pIn + 4 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 4);

#define SUMM_3x3REG_6OUT(pIn, inOffs,arIn, arW, arO) \
         LOAD_3REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_3REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 1); \
         LOAD_3REG(pIn + 2 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 2); \
         LOAD_3REG(pIn + 3 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 3); \
         LOAD_3REG(pIn + 4 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 4); \
         LOAD_3REG(pIn + 5 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 5);

#define SUMM_3x3REG_7OUT(pIn, inOffs,arIn, arW, arO) \
         LOAD_3REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_3REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 1); \
         LOAD_3REG(pIn + 2 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 2); \
         LOAD_3REG(pIn + 3 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 3); \
         LOAD_3REG(pIn + 4 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 4); \
         LOAD_3REG(pIn + 5 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 5); \
         LOAD_3REG(pIn + 6 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 6);

#define SUMM_3x3REG_8OUT(pIn, inOffs,arIn, arW, arO) \
         LOAD_3REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_3REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 1); \
         LOAD_3REG(pIn + 2 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 2); \
         LOAD_3REG(pIn + 3 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 3); \
         LOAD_3REG(pIn + 4 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 4); \
         LOAD_3REG(pIn + 5 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 5); \
         LOAD_3REG(pIn + 6 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 6); \
         LOAD_3REG(pIn + 7 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 7);

#define SUMM_3x3REG_9OUT(pIn, inOffs,arIn, arW, arO) \
         LOAD_3REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_3REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 1); \
         LOAD_3REG(pIn + 2 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 2); \
         LOAD_3REG(pIn + 3 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 3); \
         LOAD_3REG(pIn + 4 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 4); \
         LOAD_3REG(pIn + 5 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 5); \
         LOAD_3REG(pIn + 6 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 6); \
         LOAD_3REG(pIn + 7 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 7); \
         LOAD_3REG(pIn + 8 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 8);

#define SUMM_5x5REG_1OUT(arIn, arW, arO) \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, arW ## 0), arO); \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 1, arW ## 1), arO); \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 2, arW ## 2), arO); \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 3, arW ## 3), arO); \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 4, arW ## 4), arO);   

#define SUMM_6x6REG_1OUT(arIn, arW, arO) \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 0, arW ## 0), arO); \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 1, arW ## 1), arO); \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 2, arW ## 2), arO); \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 3, arW ## 3), arO); \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 4, arW ## 4), arO); \
         arO = _mm256_add_ps(_mm256_mul_ps(arIn ## 5, arW ## 5), arO); 

#define SUMM_6x6REG_2OUT(pIn, inOffs, arIn, arW, arO) \
         LOAD_6REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_6x6REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_6REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_6x6REG_1OUT(arIn, arW, arO ## 1);

#define SUMM_6x6REG_3OUT(pIn, inOffs, arIn, arW, arO) \
         LOAD_6REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_6x6REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_6REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_6x6REG_1OUT(arIn, arW, arO ## 1); \
         LOAD_6REG(pIn + 2 * inOffs, 8, arIn);  \
         SUMM_6x6REG_1OUT(arIn, arW, arO ## 2);

#define SUMM_6x6REG_4OUT(pIn, inOffs, arIn, arW, arO) \
         LOAD_6REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_6x6REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_6REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_6x6REG_1OUT(arIn, arW, arO ## 1); \
         LOAD_6REG(pIn + 2 * inOffs, 8, arIn);  \
         SUMM_6x6REG_1OUT(arIn, arW, arO ## 2); \
         LOAD_6REG(pIn + 3 * inOffs, 8, arIn);  \
         SUMM_6x6REG_1OUT(arIn, arW, arO ## 3);
    
#define SUMM_3x3REG_10OUT(pIn, inOffs, arIn, arW, arO) \
         LOAD_3REG(pIn + 0 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 0); \
         LOAD_3REG(pIn + 1 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 1); \
         LOAD_3REG(pIn + 2 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 2); \
         LOAD_3REG(pIn + 3 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 3); \
         LOAD_3REG(pIn + 4 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 4); \
         LOAD_3REG(pIn + 5 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 5); \
         LOAD_3REG(pIn + 6 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 6); \
         LOAD_3REG(pIn + 7 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 7); \
         LOAD_3REG(pIn + 8 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 8); \
         LOAD_3REG(pIn + 9 * inOffs, 8, arIn);  \
         SUMM_3x3REG_1OUT(arIn, arW, arO ## 9);


#define SUMM_REG(in, inOffs, arIn, arW, arO) \
        LOAD_REG(in, inOffs, arIn); arO = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO);
  
#define SUMM_1REG(in, inOffs, arIn, arW, arO) \
        LOAD_REG(in, inOffs, arIn); arO ## 0 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 0);

#define SUMM_2REG(in, inOffs, arIn, arW, arO) \
         LOAD_REG(in, 0 * inOffs, arIn); arO ## 0 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 0); \
         LOAD_REG(in, 1 * inOffs, arIn); arO ## 1 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 1); 

#define SUMM_3REG(in, inOffs, arIn, arW, arO) \
         LOAD_REG(in, 0 * inOffs, arIn); arO ## 0 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 0); \
         LOAD_REG(in, 1 * inOffs, arIn); arO ## 1 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 1); \
         LOAD_REG(in, 2 * inOffs, arIn); arO ## 2 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 2); 

#define SUMM_4REG(in, inOffs, arIn, arW, arO) \
         LOAD_REG(in, 0 * inOffs, arIn); arO ## 0 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 0); \
         LOAD_REG(in, 1 * inOffs, arIn); arO ## 1 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 1); \
         LOAD_REG(in, 2 * inOffs, arIn); arO ## 2 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 2); \
         LOAD_REG(in, 3 * inOffs, arIn); arO ## 3 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 3); 

#define SUMM_5REG(in, inOffs, arIn, arW, arO) \
         LOAD_REG(in, 0 * inOffs, arIn); arO ## 0 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 0); \
         LOAD_REG(in, 1 * inOffs, arIn); arO ## 1 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 1); \
         LOAD_REG(in, 2 * inOffs, arIn); arO ## 2 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 2); \
         LOAD_REG(in, 3 * inOffs, arIn); arO ## 3 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 3); \
         LOAD_REG(in, 4 * inOffs, arIn); arO ## 4 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 4); 

#define SUMM_6REG(in, inOffs, arIn, arW, arO) \
         LOAD_REG(in, 0 * inOffs, arIn); arO ## 0 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 0); \
         LOAD_REG(in, 1 * inOffs, arIn); arO ## 1 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 1); \
         LOAD_REG(in, 2 * inOffs, arIn); arO ## 2 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 2); \
         LOAD_REG(in, 3 * inOffs, arIn); arO ## 3 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 3); \
         LOAD_REG(in, 4 * inOffs, arIn); arO ## 4 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 4); \
         LOAD_REG(in, 5 * inOffs, arIn); arO ## 5 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 5); 

#define SUMM_7REG(in, inOffs, arIn, arW, arO) \
         LOAD_REG(in, 0 * inOffs, arIn); arO ## 0 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 0); \
         LOAD_REG(in, 1 * inOffs, arIn); arO ## 1 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 1); \
         LOAD_REG(in, 2 * inOffs, arIn); arO ## 2 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 2); \
         LOAD_REG(in, 3 * inOffs, arIn); arO ## 3 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 3); \
         LOAD_REG(in, 4 * inOffs, arIn); arO ## 4 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 4); \
         LOAD_REG(in, 5 * inOffs, arIn); arO ## 5 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 5); \
         LOAD_REG(in, 6 * inOffs, arIn); arO ## 6 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 6); 

#define SUMM_8REG(in, inOffs, arIn, arW, arO) \
         LOAD_REG(in, 0 * inOffs, arIn); arO ## 0 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 0); \
         LOAD_REG(in, 1 * inOffs, arIn); arO ## 1 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 1); \
         LOAD_REG(in, 2 * inOffs, arIn); arO ## 2 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 2); \
         LOAD_REG(in, 3 * inOffs, arIn); arO ## 3 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 3); \
         LOAD_REG(in, 4 * inOffs, arIn); arO ## 4 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 4); \
         LOAD_REG(in, 5 * inOffs, arIn); arO ## 5 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 5); \
         LOAD_REG(in, 6 * inOffs, arIn); arO ## 6 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 6); \
         LOAD_REG(in, 7 * inOffs, arIn); arO ## 7 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 7); 

#define SUMM_9REG(in, inOffs, arIn, arW, arO) \
         LOAD_REG(in, 0 * inOffs, arIn); arO ## 0 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 0); \
         LOAD_REG(in, 1 * inOffs, arIn); arO ## 1 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 1); \
         LOAD_REG(in, 2 * inOffs, arIn); arO ## 2 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 2); \
         LOAD_REG(in, 3 * inOffs, arIn); arO ## 3 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 3); \
         LOAD_REG(in, 4 * inOffs, arIn); arO ## 4 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 4); \
         LOAD_REG(in, 5 * inOffs, arIn); arO ## 5 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 5); \
         LOAD_REG(in, 6 * inOffs, arIn); arO ## 6 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 6); \
         LOAD_REG(in, 7 * inOffs, arIn); arO ## 7 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 7); \
         LOAD_REG(in, 8 * inOffs, arIn); arO ## 8 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 8); 

#define SUMM_10REG(in, inOffs, arIn, arW, arO) \
         LOAD_REG(in, 0 * inOffs, arIn); arO ## 0 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 0); \
         LOAD_REG(in, 1 * inOffs, arIn); arO ## 1 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 1); \
         LOAD_REG(in, 2 * inOffs, arIn); arO ## 2 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 2); \
         LOAD_REG(in, 3 * inOffs, arIn); arO ## 3 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 3); \
         LOAD_REG(in, 4 * inOffs, arIn); arO ## 4 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 4); \
         LOAD_REG(in, 5 * inOffs, arIn); arO ## 5 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 5); \
         LOAD_REG(in, 6 * inOffs, arIn); arO ## 6 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 6); \
         LOAD_REG(in, 7 * inOffs, arIn); arO ## 7 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 7); \
         LOAD_REG(in, 8 * inOffs, arIn); arO ## 8 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 8); \
         LOAD_REG(in, 9 * inOffs, arIn); arO ## 9 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 9); 

#define SUMM_11REG(in, inOffs, arIn, arW, arO) \
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
         LOAD_REG(in, 10 * inOffs, arIn); arO ## 10 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 10); 

#define SUMM_12REG(in, inOffs, arIn, arW, arO) \
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
         LOAD_REG(in, 11 * inOffs, arIn); arO ## 11 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 11); 

#define SUMM_13REG(in, inOffs, arIn, arW, arO) \
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
         LOAD_REG(in, 12 * inOffs, arIn); arO ## 12 = _mm256_add_ps(_mm256_mul_ps(arIn, arW), arO ## 12); 

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


#define SET_OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO);

#define SET_1OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0);

#define SET_2OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1);

#define SET_3OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2);

#define SET_4OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2); \
    pOut[3] = bias + horSummReg<__m256>(arO ## 3);

#define SET_5OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2); \
    pOut[3] = bias + horSummReg<__m256>(arO ## 3); \
    pOut[4] = bias + horSummReg<__m256>(arO ## 4);

#define SET_6OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2); \
    pOut[3] = bias + horSummReg<__m256>(arO ## 3); \
    pOut[4] = bias + horSummReg<__m256>(arO ## 4); \
    pOut[5] = bias + horSummReg<__m256>(arO ## 5);

#define SET_7OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2); \
    pOut[3] = bias + horSummReg<__m256>(arO ## 3); \
    pOut[4] = bias + horSummReg<__m256>(arO ## 4); \
    pOut[5] = bias + horSummReg<__m256>(arO ## 5); \
    pOut[6] = bias + horSummReg<__m256>(arO ## 6);
    
#define SET_8OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2); \
    pOut[3] = bias + horSummReg<__m256>(arO ## 3); \
    pOut[4] = bias + horSummReg<__m256>(arO ## 4); \
    pOut[5] = bias + horSummReg<__m256>(arO ## 5); \
    pOut[6] = bias + horSummReg<__m256>(arO ## 6); \
    pOut[7] = bias + horSummReg<__m256>(arO ## 7);

#define SET_9OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2); \
    pOut[3] = bias + horSummReg<__m256>(arO ## 3); \
    pOut[4] = bias + horSummReg<__m256>(arO ## 4); \
    pOut[5] = bias + horSummReg<__m256>(arO ## 5); \
    pOut[6] = bias + horSummReg<__m256>(arO ## 6); \
    pOut[7] = bias + horSummReg<__m256>(arO ## 7); \
    pOut[8] = bias + horSummReg<__m256>(arO ## 8);

#define SET_10OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2); \
    pOut[3] = bias + horSummReg<__m256>(arO ## 3); \
    pOut[4] = bias + horSummReg<__m256>(arO ## 4); \
    pOut[5] = bias + horSummReg<__m256>(arO ## 5); \
    pOut[6] = bias + horSummReg<__m256>(arO ## 6); \
    pOut[7] = bias + horSummReg<__m256>(arO ## 7); \
    pOut[8] = bias + horSummReg<__m256>(arO ## 8); \
    pOut[9] = bias + horSummReg<__m256>(arO ## 9);

#define SET_11OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2); \
    pOut[3] = bias + horSummReg<__m256>(arO ## 3); \
    pOut[4] = bias + horSummReg<__m256>(arO ## 4); \
    pOut[5] = bias + horSummReg<__m256>(arO ## 5); \
    pOut[6] = bias + horSummReg<__m256>(arO ## 6); \
    pOut[7] = bias + horSummReg<__m256>(arO ## 7); \
    pOut[8] = bias + horSummReg<__m256>(arO ## 8); \
    pOut[9] = bias + horSummReg<__m256>(arO ## 9); \
    pOut[10] = bias + horSummReg<__m256>(arO ## 10);

#define SET_12OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2); \
    pOut[3] = bias + horSummReg<__m256>(arO ## 3); \
    pOut[4] = bias + horSummReg<__m256>(arO ## 4); \
    pOut[5] = bias + horSummReg<__m256>(arO ## 5); \
    pOut[6] = bias + horSummReg<__m256>(arO ## 6); \
    pOut[7] = bias + horSummReg<__m256>(arO ## 7); \
    pOut[8] = bias + horSummReg<__m256>(arO ## 8); \
    pOut[9] = bias + horSummReg<__m256>(arO ## 9); \
    pOut[10] = bias + horSummReg<__m256>(arO ## 10); \
    pOut[11] = bias + horSummReg<__m256>(arO ## 11);

#define SET_13OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2); \
    pOut[3] = bias + horSummReg<__m256>(arO ## 3); \
    pOut[4] = bias + horSummReg<__m256>(arO ## 4); \
    pOut[5] = bias + horSummReg<__m256>(arO ## 5); \
    pOut[6] = bias + horSummReg<__m256>(arO ## 6); \
    pOut[7] = bias + horSummReg<__m256>(arO ## 7); \
    pOut[8] = bias + horSummReg<__m256>(arO ## 8); \
    pOut[9] = bias + horSummReg<__m256>(arO ## 9); \
    pOut[10] = bias + horSummReg<__m256>(arO ## 10); \
    pOut[11] = bias + horSummReg<__m256>(arO ## 11); \
    pOut[12] = bias + horSummReg<__m256>(arO ## 12);

#define SET_14OUT(arO, pOut) \
    pOut[0] = bias + horSummReg<__m256>(arO ## 0); \
    pOut[1] = bias + horSummReg<__m256>(arO ## 1); \
    pOut[2] = bias + horSummReg<__m256>(arO ## 2); \
    pOut[3] = bias + horSummReg<__m256>(arO ## 3); \
    pOut[4] = bias + horSummReg<__m256>(arO ## 4); \
    pOut[5] = bias + horSummReg<__m256>(arO ## 5); \
    pOut[6] = bias + horSummReg<__m256>(arO ## 6); \
    pOut[7] = bias + horSummReg<__m256>(arO ## 7); \
    pOut[8] = bias + horSummReg<__m256>(arO ## 8); \
    pOut[9] = bias + horSummReg<__m256>(arO ## 9); \
    pOut[10] = bias + horSummReg<__m256>(arO ## 10); \
    pOut[11] = bias + horSummReg<__m256>(arO ## 11); \
    pOut[12] = bias + horSummReg<__m256>(arO ## 12); \
    pOut[13] = bias + horSummReg<__m256>(arO ## 13);


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

    template<size_t M, size_t S, size_t D, size_t RO>
    void reorderInputCHW2HCW(const SN_Base::snSize& insz, const SN_Base::snFloat* input, const SN_Base::snSize& outsz, SN_Base::snFloat* output){

        SN_Base::snFloat* pOut = output;
       
        if (M == 1){

            for (size_t i = 0; i < (outsz.w * outsz.h) / RO; ++i){

                for (size_t j = 0; j < insz.d / 8; ++j){

                    for (size_t k = 0; k < RO; ++k){

                        for (size_t t = 0; t < 8; ++t){

                            size_t ci = (i * RO + k) % outsz.w, cr = (i * RO + k) / outsz.w;

                            const snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * (t + j * 8);

                            *pOut = *pIn;

                            pOut += M * (M - 1);
                        }
                    }
                }

                if (insz.d % 8){
                    for (size_t k = 0; k < RO; ++k){

                        for (size_t t = 0; t < insz.d % 8; ++t){

                            size_t ci = (i * RO + k) % outsz.w, cr = (i * RO + k) / outsz.w;

                            const snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * (t + (insz.d / 8) * 8);

                            *pOut = *pIn;

                            pOut += M * (M - 1);
                        }
                    }
                }
            }

            const size_t rmr = (outsz.w * outsz.h) % RO;
            if (rmr){

                const size_t offs = ((outsz.w * outsz.h) / RO) * RO;

                for (size_t j = 0; j < insz.d / 8; ++j){

                    for (size_t k = 0; k < rmr; ++k){

                        for (size_t t = 0; t < 8; ++t){

                            size_t ci = (offs + k) % outsz.w, cr = (offs + k) / outsz.w;

                            const snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * (t + j * 8);

                            *pOut = *pIn;

                            pOut += M * (M - 1);
                        }
                    }
                }
                
                if (insz.d % 8){
                    for (size_t k = 0; k < rmr; ++k){

                        for (size_t t = 0; t < insz.d % 8; ++t){

                            size_t ci = (offs + k) % outsz.w, cr = (offs + k) / outsz.w;

                            const snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * (t + (insz.d / 8) * 8);

                            *pOut = *pIn;

                            pOut += M * (M - 1);
                        }
                    }
                }
            }
        }

        /////////////////////////////////////////////

        else if ((M == 3) && (D == 1)){

            for (size_t i = 0; i < (outsz.w * outsz.h) / RO; ++i){
                             
                for (size_t j = 0;  j < insz.d; ++j){

                    for (size_t k = 0; k < RO; ++k){

                        size_t ci = (i * RO + k) % outsz.w, cr = (i * RO + k) / outsz.w;

                        const snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * j;

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));   

                        pOut += M * (M - 1);
                    }               
                }
            }
                       
            const size_t rmr = (outsz.w * outsz.h) % RO;
            if (rmr){

                const size_t offs = ((outsz.w * outsz.h) / RO) * RO;

                for (size_t j = 0; j < insz.d; ++j){

                    for (size_t k = 0; k < rmr; ++k){

                        size_t ci = (offs + k) % outsz.w, cr = (offs + k) / outsz.w;

                        const snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * j;

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));

                        pOut += M * (M - 1);
                    }
                }
            }
        }

        /////////////////////////////////////////////

        else if ((M == 5) && (D == 1)){

            for (size_t i = 0; i < (outsz.w * outsz.h) / RO; ++i){

                for (size_t j = 0; j < insz.d; ++j){

                    for (size_t k = 0; k < RO; ++k){

                        size_t ci = (i * RO + k) % outsz.w, cr = (i * RO + k) / outsz.w;

                        const snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * j;

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                        _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                        _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));

                        pOut += M * (M - 1);
                    }
                }
            }

            const size_t rmr = (outsz.w * outsz.h) % RO;
            if (rmr){

                const size_t offs = ((outsz.w * outsz.h) / RO) * RO;

                for (size_t j = 0; j < insz.d; ++j){

                    for (size_t k = 0; k < rmr; ++k){

                        size_t ci = (offs + k) % outsz.w, cr = (offs + k) / outsz.w;

                        const snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * j;

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                        _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                        _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));

                        pOut += M * (M - 1);
                    }
                }
            }
        }

        /////////////////////////////////////////////

        else if ((M == 7) && (D == 1)){

            for (size_t i = 0; i < (outsz.w * outsz.h) / RO; ++i){

                for (size_t j = 0; j < insz.d; ++j){

                    for (size_t k = 0; k < RO; ++k){

                        size_t ci = (i * RO + k) % outsz.w, cr = (i * RO + k) / outsz.w;

                        const snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * j;

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                        _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                        _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));
                        _mm256_storeu_ps(pOut + 5 * M, _mm256_loadu_ps(pIn + 5 * insz.w));
                        _mm256_storeu_ps(pOut + 6 * M, _mm256_loadu_ps(pIn + 6 * insz.w));

                        pOut += M * (M - 1);
                    }
                }
            }

            const size_t rmr = (outsz.w * outsz.h) % RO;
            if (rmr){

                const size_t offs = ((outsz.w * outsz.h) / RO) * RO;

                for (size_t j = 0; j < insz.d; ++j){

                    for (size_t k = 0; k < rmr; ++k){

                        size_t ci = (offs + k) % outsz.w, cr = (offs + k) / outsz.w;

                        const snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * j;

                        _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                        _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                        _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                        _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                        _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));
                        _mm256_storeu_ps(pOut + 5 * M, _mm256_loadu_ps(pIn + 5 * insz.w));
                        _mm256_storeu_ps(pOut + 6 * M, _mm256_loadu_ps(pIn + 6 * insz.w));

                        pOut += M * (M - 1);
                    }
                }
            }
        }

        /////////////////////////////////////////////

        else if ((M == 9) && (D == 1)){  // RO == 1

            for (size_t i = 0; i < outsz.h; ++i){

                for (size_t j = 0; j < outsz.w; ++j){

                    const snFloat* pIn = input + S * insz.w * i + S * j;

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
                     
                        pIn += insz.w * insz.h;
                        pOut += M * (M - 1);
                    }
                }
            }
        }             

        ///////////////////////////////////////////////////
        

        else if ((M == 3) && (D == 2)){
          
        }

        else if ((M == 5) && (D == 2)){

        }

        else if ((M == 7) && (D == 2)){

        }

        else if ((M == 9) && (D == 2)){

            
        }
    };

    template<size_t M>
    void reorderWeight9To8(const SN_Base::snSize& insz, const SN_Base::snFloat* input, const SN_Base::snSize& outsz, SN_Base::snFloat* output){

        SN_Base::snFloat* pOut = output,
                        * pEnd = output;

        if (M == 1){



        }

        else if (M == 3){



        }
    }

    template<size_t M, size_t RO>
    void getPeakOutput(size_t W, const SN_Base::snFloat* pIn, const SN_Base::snFloat* pW, SN_Base::snFloat* pOut){
                
        for (size_t i = 0; i < W; ++i){

            for (size_t j = 0; j < RO; ++j)
              pOut[j] += pIn[M * M * j + (M * M - 1) + i * RO * M * M] * pW[M * M * i + (M * M - 1)];
        }
    }       
};