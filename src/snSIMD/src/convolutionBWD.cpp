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
      
    template <size_t M>
    void backwardGW(size_t stride, size_t dilate,
        const snFloat* weight, const snSize& insz, const snFloat* input, const snSize& outsz, const snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){

        const size_t kernel = outsz.d,
                     wStepByD = M * M,                      // step weight by input
                     wStepByK = wStepByD * insz.d,          // step weight by output
                     wStepByN = wStepByK * kernel + kernel, // step weight by batch
                     inStepByD = insz.w * insz.h,           // step in by input
                     inStepByN = inStepByD * insz.d,        // step in by batch
                     outStepByD = outsz.w * outsz.h,        // step out by input
                     outStepByN = outStepByD * outsz.d;     // step out by batch

        snFloat* wgThr = (insz.n == 1) ? dWeightOut : (snFloat*)calloc(wStepByN * insz.n, sizeof(snFloat));

        memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));
        memset(dWeightOut, 0, wStepByN * sizeof(snFloat));

        auto core = std::thread::hardware_concurrency();
        if (core == 0) core = 4;

        // by batch
#pragma omp parallel for num_threads(core)
        for (int n = 0; n < int(insz.n); ++n){

            snFloat* wBuff = wgThr + wStepByN * n;
            __m256 arGOut[wStepByD / 8];
            snFloat mGOut[wStepByD], In[wStepByD], W[wStepByD];

            for (size_t p = 0; p < outStepByD; ++p){

                size_t ox = p % outsz.w, oy = p / outsz.w,
                    posW = ox * stride, posH = oy * stride;

                const snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;
                   
                snFloat* pdW = wBuff + wStepByK * kernel;

                // on all out layers
                for (size_t k = 0; k < kernel; ++k){
                    *(pdW + k) += pGrIn[k * outStepByD];      // + bias
                }

                // on all in layers               
                for (size_t d = 0; d < insz.d; ++d){

                    const snFloat* pIn = input + inStepByD * d + inStepByN * n;
                    snFloat* pGrOut = gradOut + inStepByD * d + inStepByN * n;

                    const snFloat* pW = weight + wStepByD * d;
                    pdW = wBuff + wStepByD * d;

                    for (int z = 0; z < wStepByD / 8; ++z)
                        arGOut[z] = _mm256_setzero_ps();

                    for (size_t c = 0; c < wStepByD; ++c){

                        size_t cx = c % M, cy = c / M;
                        In[c] = *(pIn + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w);
                    }

                    // on all out layers
                    for (size_t k = 0; k < kernel; ++k){

                        for (size_t c = 0; c < wStepByD; ++c){

                            size_t cx = c % M, cy = c / M;
                            W[c] = *(pW + cx + cy * M);
                        }

                        snFloat gin = pGrIn[k * outStepByD];

                        __m256 arGIn = _mm256_set1_ps(gin);

                        for (int z = 0; z < wStepByD / 8; ++z){

                            __m256 arW = _mm256_loadu_ps(W + z * 8);

                            arGOut[z] = _mm256_add_ps(arGOut[z], _mm256_mul_ps(arGIn, arW));

                            __m256 arIn = _mm256_loadu_ps(In + z * 8);

                            _mm256_storeu_ps(W + z * 8, _mm256_mul_ps(arGIn, arIn));
                        }

#define DW(c, r)   *(pdW + (c) + (r) * M)

                        for (size_t c = 0; c < (wStepByD - 1); ++c){

                            size_t cx = c % M, cy = c / M;

                            DW(cx, cy) += W[c];
                        }
                        DW(M - 1, M - 1) += gin * In[wStepByD - 1];


#define GOut(c, r) *(pGrOut + ((c) + posW + (c) * (dilate - 1)) + ((r) + posH + (r) * (dilate - 1)) * insz.w)

                        GOut(M - 1, M - 1) += gin * W[wStepByD - 1];

                        pW += wStepByK;
                        pdW += wStepByK;
                    }

                    for (int z = 0; z < wStepByD / 8; ++z)
                        _mm256_storeu_ps(mGOut + z * 8, arGOut[z]);

                    for (size_t c = 0; c < (wStepByD - 1); ++c){

                        size_t cx = c % M, cy = c / M;

                        GOut(cx, cy) += mGOut[c];
                    }
                }
            }
        }

        if (insz.n > 1){
            for (size_t i = 0; i < insz.n; ++i){
                snFloat* wBuff = wgThr + wStepByN * i;
                for (size_t j = 0; j < wStepByN; ++j)
                    dWeightOut[j] += wBuff[j];
            }
            for (size_t j = 0; j < wStepByN; ++j)
                dWeightOut[j] /= insz.n;

            free(wgThr);
        }

#undef GOut
#undef DW

    }

    template <size_t M>
    void backwardG(size_t stride, size_t dilate,
        const snFloat* weight, const snSize& insz, const snSize& outsz, const snFloat* gradIn, snFloat* gradOut){

        const size_t kernel = outsz.d,
                     wStepByD = M * M,                      // step weight by input
                     wStepByK = wStepByD * insz.d,          // step weight by output
                     wStepByN = wStepByK * kernel + kernel, // step weight by batch
                     inStepByD = insz.w * insz.h,           // step in by input
                     inStepByN = inStepByD * insz.d,        // step in by batch
                     outStepByD = outsz.w * outsz.h,        // step out by input
                     outStepByN = outStepByD * outsz.d;     // step out by batch

        memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));

        auto core = std::thread::hardware_concurrency();
        if (core == 0) core = 4;

        // by batch
#pragma omp parallel for num_threads(core)
        for (int n = 0; n < int(insz.n); ++n){

            __m256 arGOut[wStepByD / 8];
            snFloat mGOut[wStepByD], W[wStepByD];

            for (size_t p = 0; p < outStepByD; ++p){

                size_t ox = p % outsz.w, oy = p / outsz.w,
                    posW = ox * stride, posH = oy * stride;

                const snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;

                // on all in layers               
                for (size_t d = 0; d < insz.d; ++d){

                    snFloat* pGrOut = gradOut + inStepByD * d + inStepByN * n;

                    const snFloat* pW = weight + wStepByD * d;

                    for (int z = 0; z < wStepByD / 8; ++z)
                        arGOut[z] = _mm256_setzero_ps();

                    // on all out layers
                    for (size_t k = 0; k < kernel; ++k){

                        for (size_t c = 0; c < wStepByD; ++c){

                            size_t cx = c % M, cy = c / M;
                            W[c] = *(pW + cx + cy * M);
                        }

                        snFloat gin = pGrIn[k * outStepByD];

                        __m256 arGIn = _mm256_set1_ps(gin);

                        for (int z = 0; z < wStepByD / 8; ++z){

                            __m256 arW = _mm256_loadu_ps(W + z * 8);

                            arGOut[z] = _mm256_add_ps(arGOut[z], _mm256_mul_ps(arGIn, arW));
                        }

#define GOut(c, r) *(pGrOut + ((c) + posW + (c) * (dilate - 1)) + ((r) + posH + (r) * (dilate - 1)) * insz.w)

                        GOut(M - 1, M - 1) += gin * W[wStepByD - 1];

                        pW += wStepByK;
                    }

                    for (int z = 0; z < wStepByD / 8; ++z)
                        _mm256_storeu_ps(mGOut + z * 8, arGOut[z]);

                    for (size_t c = 0; c < (wStepByD - 1); ++c){

                        size_t cx = c % M, cy = c / M;

                        GOut(cx, cy) += mGOut[c];
                    }
                }
            }
        }

#undef GOut

    }
    
    bool convolutionBWD_GW(size_t M, size_t S, size_t D,
        const snFloat* weight,
        const snSize& insz, const snFloat* input,
        const snSize& outsz, const snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){
   
#define dbwd(MS)   \
    if (M == MS){  \
        backwardGW<MS>(S, D, weight, insz, input, outsz, gradIn, gradOut, dWeightOut); return true; };
                
        dbwd(3)
        dbwd(5)
        dbwd(7)
        dbwd(9)

        return false;

#undef dbwd
    };
    
    bool convolutionBWD_G(size_t M, size_t S, size_t D,
        const snFloat* weight, const snSize& insz, const snSize& outsz, const snFloat* gradIn, snFloat* gradOut){

#define dbwd(MS)   \
    if (M == MS){  \
        backwardG<MS>(S, D, weight, insz, outsz, gradIn, gradOut); return true; };
                
        dbwd(3)
        dbwd(5)
        dbwd(7)
        dbwd(9)

        return false;

#undef dbwd
    };
};

