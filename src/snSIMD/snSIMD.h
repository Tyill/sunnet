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

using namespace SN_Base;
using namespace std;

namespace SN_SIMD{

    const int LAYER_MAX_WIDTH = 800;
    const int LAYER_MAX_HEIGHT = 600;
    const int REG_CNT = 16;                   // registr count
    const int REG_BYTE_SZ = 32;               // registr byte size  (256 bit = 32 kB)
    const int L1_BYTE_SZ = 32 * 1024;         // L1 cache byte size (32 kB)
    const int L2_BYTE_SZ = 256 * 1024;        // L2 cache byte size (256 kB)
    const int L3_BYTE_SZ = 8 * 1024 * 1024;   // L3 cache byte size (8 MB)

#define LOAD_REG(in, reg)  __m256 reg = _mm256_loadu_ps(in);
#define LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(in, buff, w, reg) buff[0] = *(in);               \
                                                           buff[1] = *(in + 1);           \
                                                           buff[2] = *(in + 2);           \
                                                           buff[3] = *(in + (w));         \
                                                           buff[4] = *(in + (w) + 1);     \
                                                           buff[5] = *(in + (w) + 2);     \
                                                           buff[6] = *(in + 2 * (w));     \
                                                           buff[7] = *(in + 2 * (w) + 1); \
                                                        __m256 reg = _mm256_loadu_ps(buff);

    float horSummAVX(__m256 a) {

        __m128 hi = _mm256_extractf128_ps(a, 1);
        __m128 lo = _mm256_extractf128_ps(a, 0);
        lo = _mm_add_ps(hi, lo);
        hi = _mm_movehl_ps(hi, lo);
        lo = _mm_add_ps(hi, lo);
        hi = _mm_shuffle_ps(lo, lo, 1);
        lo = _mm_add_ss(hi, lo);
        return _mm_cvtss_f32(lo);
    }

    void reorder3x3Stride1Dilate1(size_t isz, snFloat* input){


    }

    void micro3x3Stride1Dilate1(snFloat* weight,
        const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

        // NCHW

        const int MASK = 3,   
                  MASK_MAX_CNT = L1_BYTE_SZ / (sizeof(snFloat) * MASK * MASK),
                  L1_SIZE = MASK_MAX_CNT * (sizeof(snFloat) * MASK * MASK);

        snFloat L1Cache[L1_SIZE];
        
        const int W = insz.w,
                  H = insz.h,
                  LAYER_SZ = W * H,
                  IN_ALL_SZ = insz.size(),
                  L1_CNT = IN_ALL_SZ / L1_SIZE;
        
        for (size_t i = 0; i < L1_CNT; ++i){

            memcpy(L1Cache, input + i * L1_SIZE, L1_SIZE * sizeof(snFloat));
           
            LOAD_REG(weight, arW);

            snFloat LBUFF[MASK * MASK];
             
            snFloat* pL1 = L1Cache;
            
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar0);  pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar1);  pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar2);  pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar3);  pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar4);  pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar5);  pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar6);  pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar7);  pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar8);  pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar9);  pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar10); pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar11); pL1 += W * H;
            LOAD_REG_FROM_BUFF_3x3_1STR_1DIL(pL1, LBUFF, W, ar12); pL1 += W * H;
                        
        }

        //        
        //
        //
        //        for (size_t p = 0; p < outStepByD; ++p){
        //
        //            size_t ox = p % outsz.w, oy = p / outsz.w,
        //                posW = ox * stride, posH = oy * stride;
        //
        //            memset(outBuff, 0, kernel * sizeof(snFloat));
        //
        //            snFloat* pIn = input + inStepByN * n;
        //            snFloat* pW = weight;
        //            
        //            // on all in layers
        //            for (size_t d = 0; d < insz.d; ++d){
        //
        //                for (size_t c = 0; c < wStepByD; ++c){
        //
        //                    size_t cx = c % R, cy = c / R;
        //                    In[c] = *(pIn + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w);
        //                }
        //
        //                pW = weight + wStepByD * d;
        //
        //                // on all out layers
        //                for (size_t k = 0; k < kernel; ++k){
        //
        //                    for (size_t c = 0; c < wStepByD; ++c){
        //
        //                        size_t cx = c % R, cy = c / R;
        //                        W[c] = *(pW + cx + cy * R);
        //                    }
        //
        //                    __m256 arOut = _mm256_setzero_ps();
        //
        //                    for (int z = 0; z < wStepByD / 8; ++z){
        //
        //                        __m256 arIn = _mm256_loadu_ps(In + z * 8);
        //
        //                        __m256 arW = _mm256_loadu_ps(W + z * 8);
        //
        //                        arOut = _mm256_add_ps(arOut, _mm256_mul_ps(arIn, arW));
        //                    }
        //
        //                    outBuff[k] += horSummAVX(arOut);
        //
        //                    outBuff[k] += In[wStepByD - 1] * W[wStepByD - 1];
        //
        //                    pW += wStepByK;
        //                }
        //
        //                pIn += inStepByD;
        //
        //            }
        //
        //            snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
        //            pW = weight + wStepByN;
        //
        //            // on all out layers
        //            for (size_t k = 0; k < kernel; ++k){
        //
        //                *pOut = outBuff[k] + *(pW + k); // + bias              
        //
        //                pOut += outStepByD;
        //            }
        //        }
        //
        //    }
        //
        //
        //    template <int R>
        //    void forward(size_t kernel, size_t stride, size_t dilate,
        //        snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){
        //
        //        const size_t wStepByD = R * R,          // step weight by input
        //            wStepByK = wStepByD * insz.d,       // step weight by output
        //            wStepByN = wStepByK * kernel,       // step weight by batch
        //            inStepByD = insz.w * insz.h,        // step in by input
        //            inStepByN = inStepByD * insz.d,     // step in by batch
        //            outStepByD = outsz.w * outsz.h,     // step out by input
        //            outStepByN = outStepByD * outsz.d;  // step out by batch
        //
        //        size_t shareStepByN = kernel;           // for local mem
        //        snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));
        //
        //        // by batch
        //#pragma omp parallel for
        //        for (int n = 0; n < int(insz.n); ++n){
        //
        //            snFloat* outBuff = share + shareStepByN * n;
        //            snFloat In[wStepByD], W[wStepByD];
        //
        //            for (size_t p = 0; p < outStepByD; ++p){
        //
        //                size_t ox = p % outsz.w, oy = p / outsz.w,
        //                    posW = ox * stride, posH = oy * stride;
        //
        //                memset(outBuff, 0, kernel * sizeof(snFloat));
        //
        //                snFloat* pIn = input + inStepByN * n;
        //                snFloat* pW = weight;
        //
        //                // on all in layers
        //                for (size_t d = 0; d < insz.d; ++d){
        //
        //                    for (size_t c = 0; c < wStepByD; ++c){
        //
        //                        size_t cx = c % R, cy = c / R;
        //                        In[c] = *(pIn + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w);
        //                    }
        //
        //                    pW = weight + wStepByD * d;
        //
        //                    // on all out layers
        //                    for (size_t k = 0; k < kernel; ++k){
        //
        //                        for (size_t c = 0; c < wStepByD; ++c){
        //
        //                            size_t cx = c % R, cy = c / R;
        //                            W[c] = *(pW + cx + cy * R);
        //                        }
        //
        //                        __m256 arOut = _mm256_setzero_ps();
        //
        //                        for (int z = 0; z < wStepByD / 8; ++z){
        //
        //                            __m256 arIn = _mm256_loadu_ps(In + z * 8);
        //
        //                            __m256 arW = _mm256_loadu_ps(W + z * 8);
        //
        //                            arOut = _mm256_add_ps(arOut, _mm256_mul_ps(arIn, arW));
        //                        }
        //
        //                        outBuff[k] += horSummAVX(arOut);
        //
        //                        outBuff[k] += In[wStepByD - 1] * W[wStepByD - 1];
        //
        //                        pW += wStepByK;
        //                    }
        //
        //                    pIn += inStepByD;
        //
        //                }
        //
        //                snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
        //                pW = weight + wStepByN;
        //
        //                // on all out layers
        //                for (size_t k = 0; k < kernel; ++k){
        //
        //                    *pOut = outBuff[k] + *(pW + k); // + bias              
        //
        //                    pOut += outStepByD;
        //                }
        //            }
        //        }
        //
        //        free(share);
        //    }
    }
}