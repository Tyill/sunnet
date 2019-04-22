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
      
    template <int R>
    void backwardGW(size_t kernel, size_t stride, size_t dilate,
        snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){

        const size_t wStepByD = R * R,              // step weight by input
            wStepByK = wStepByD * insz.d,          // step weight by output
            wStepByN = wStepByK * kernel + kernel, // step weight by batch
            inStepByD = insz.w * insz.h,           // step in by input
            inStepByN = inStepByD * insz.d,        // step in by batch
            outStepByD = outsz.w * outsz.h,        // step out by input
            outStepByN = outStepByD * outsz.d;     // step out by batch

        snFloat* wgThr = (insz.n == 1) ? dWeightOut : (snFloat*)calloc(wStepByN * insz.n, sizeof(snFloat));

        memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));
        memset(dWeightOut, 0, wStepByN * sizeof(snFloat));


        // by batch
#pragma omp parallel for
        for (int n = 0; n < int(insz.n); ++n){

            snFloat* wBuff = wgThr + wStepByN * n;
            __m256 arGOut[wStepByD / 8];
            snFloat mGOut[wStepByD], In[wStepByD], W[wStepByD];

            for (size_t p = 0; p < outStepByD; ++p){

                size_t ox = p % outsz.w, oy = p / outsz.w,
                    posW = ox * stride, posH = oy * stride;

                snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;
                snFloat* pdW = wBuff + wStepByK * kernel;

                // on all out layers
                for (size_t k = 0; k < kernel; ++k){
                    *(pdW + k) += pGrIn[k * outStepByD];      // + bias
                }

                // on all in layers               
                for (size_t d = 0; d < insz.d; ++d){

                    snFloat* pIn = input + inStepByD * d + inStepByN * n;
                    snFloat* pGrOut = gradOut + inStepByD * d + inStepByN * n;

                    snFloat* pW = weight + wStepByD * d;
                    pdW = wBuff + wStepByD * d;

                    for (int z = 0; z < wStepByD / 8; ++z)
                        arGOut[z] = _mm256_setzero_ps();

                    for (size_t c = 0; c < wStepByD; ++c){

                        size_t cx = c % R, cy = c / R;
                        In[c] = *(pIn + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w);
                    }

                    // on all out layers
                    for (size_t k = 0; k < kernel; ++k){

                        for (size_t c = 0; c < wStepByD; ++c){

                            size_t cx = c % R, cy = c / R;
                            W[c] = *(pW + cx + cy * R);
                        }

                        snFloat gin = pGrIn[k * outStepByD];

                        __m256 arGIn = _mm256_set1_ps(gin);

                        for (int z = 0; z < wStepByD / 8; ++z){

                            __m256 arW = _mm256_loadu_ps(W + z * 8);

                            arGOut[z] = _mm256_add_ps(arGOut[z], _mm256_mul_ps(arGIn, arW));

                            __m256 arIn = _mm256_loadu_ps(In + z * 8);

                            _mm256_storeu_ps(W + z * 8, _mm256_mul_ps(arGIn, arIn));
                        }

#define DW(c, r)   *(pdW + (c) + (r) * R)

                        for (size_t c = 0; c < (wStepByD - 1); ++c){

                            size_t cx = c % R, cy = c / R;

                            DW(cx, cy) += W[c];
                        }
                        DW(R - 1, R - 1) += gin * In[wStepByD - 1];


#define GOut(c, r) *(pGrOut + ((c) + posW + (c) * (dilate - 1)) + ((r) + posH + (r) * (dilate - 1)) * insz.w)

                        GOut(R - 1, R - 1) += gin * W[wStepByD - 1];

                        pW += wStepByK;
                        pdW += wStepByK;
                    }

                    for (int z = 0; z < wStepByD / 8; ++z)
                        _mm256_storeu_ps(mGOut + z * 8, arGOut[z]);

                    for (size_t c = 0; c < (wStepByD - 1); ++c){

                        size_t cx = c % R, cy = c / R;

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

    template <int R>
    void backwardG(size_t kernel, size_t stride, size_t dilate,
        snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut){

        const size_t wStepByD = R * R,             // step weight by input
            wStepByK = wStepByD * insz.d,          // step weight by output
            wStepByN = wStepByK * kernel + kernel, // step weight by batch
            inStepByD = insz.w * insz.h,           // step in by input
            inStepByN = inStepByD * insz.d,        // step in by batch
            outStepByD = outsz.w * outsz.h,        // step out by input
            outStepByN = outStepByD * outsz.d;     // step out by batch

        memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));

        // by batch
#pragma omp parallel for
        for (int n = 0; n < int(insz.n); ++n){

            __m256 arGOut[wStepByD / 8];
            snFloat mGOut[wStepByD], W[wStepByD];

            for (size_t p = 0; p < outStepByD; ++p){

                size_t ox = p % outsz.w, oy = p / outsz.w,
                    posW = ox * stride, posH = oy * stride;

                snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;

                // on all in layers               
                for (size_t d = 0; d < insz.d; ++d){

                    snFloat* pGrOut = gradOut + inStepByD * d + inStepByN * n;

                    snFloat* pW = weight + wStepByD * d;

                    for (int z = 0; z < wStepByD / 8; ++z)
                        arGOut[z] = _mm256_setzero_ps();

                    // on all out layers
                    for (size_t k = 0; k < kernel; ++k){

                        for (size_t c = 0; c < wStepByD; ++c){

                            size_t cx = c % R, cy = c / R;
                            W[c] = *(pW + cx + cy * R);
                        }

                        snFloat gin = pGrIn[k * outStepByD];

                        __m256 arGIn = _mm256_set1_ps(gin);

                        for (int z = 0; z < wStepByD / 8; ++z){

                            __m256 arW = _mm256_loadu_ps(W + z * 8);

                            arGOut[z] = _mm256_add_ps(arGOut[z], _mm256_mul_ps(arGIn, arW));
                        }

#define GOut(c, r) *(pGrOut + ((c) + posW + (c) * (dilate - 1)) + ((r) + posH + (r) * (dilate - 1)) * insz.w)

                        GOut(R - 1, R - 1) += gin * W[wStepByD - 1];

                        pW += wStepByK;
                    }

                    for (int z = 0; z < wStepByD / 8; ++z)
                        _mm256_storeu_ps(mGOut + z * 8, arGOut[z]);

                    for (size_t c = 0; c < (wStepByD - 1); ++c){

                        size_t cx = c % R, cy = c / R;

                        GOut(cx, cy) += mGOut[c];
                    }
                }
            }
        }

#undef GOut

    }


    bool convolutionBWD(size_t M, size_t S, size_t D,
        snFloat* weight,
        const snSize& insz, snFloat* input,
        const snSize& outsz, snFloat* output){

        return false;


    };
};

