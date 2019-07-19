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
#include <iostream>
#include "snBase/snBase.h"
#include "Lib/OpenBLAS/cblas.h"

using namespace std;
using namespace SN_Base;


namespace SN_SIMD{
      
    template<typename T>
    float horSummReg(T a);

    template<>
    inline float horSummReg<__m256>(__m256 a){

        __m128 hi = _mm256_extractf128_ps(a, 1);
        __m128 lo = _mm256_extractf128_ps(a, 0);
        lo = _mm_add_ps(hi, lo);
        hi = _mm_movehl_ps(hi, lo);
        lo = _mm_add_ps(hi, lo);
        hi = _mm_shuffle_ps(lo, lo, 1);
        lo = _mm_add_ss(hi, lo);
        return _mm_cvtss_f32(lo);
    };

    template<size_t M>
    void reorderInputCHW2HCW(size_t S, const SN_Base::snSize& insz, const SN_Base::snFloat* input, const SN_Base::snSize& outsz, SN_Base::snFloat* output){

        SN_Base::snFloat* pOut = output;

        if (M == 1){

            for (size_t i = 0; i < (outsz.w * outsz.h); ++i){

                for (size_t j = 0; j < insz.d; ++j){

                    size_t ci = i % outsz.w, cr = i / outsz.w;                   
                 
                    *pOut = *(input + S * insz.w * cr + S * ci + insz.w * insz.h * j);

                    ++pOut;
                }
            }
        }

        /////////////////////////////////////////////

        else if (M == 3){

            for (size_t i = 0; i < (outsz.w * outsz.h); ++i){

                for (size_t j = 0; j < insz.d; ++j){

                    size_t ci = i % outsz.w, cr = i / outsz.w;

                    const SN_Base::snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * j;

                    _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                    _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                    _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));

                    pOut += M * M;
                }
            }
        }

        /////////////////////////////////////////////

        else if (M == 5){

            for (size_t i = 0; i < (outsz.w * outsz.h); ++i){

                for (size_t j = 0; j < insz.d; ++j){

                    size_t ci = i % outsz.w, cr = i / outsz.w;

                    const SN_Base::snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * j;

                    _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                    _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                    _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                    _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                    _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));

                    pOut += M * M;

                }
            }
        }

        /////////////////////////////////////////////

        else if (M == 7){

            for (size_t i = 0; i < (outsz.w * outsz.h); ++i){

                for (size_t j = 0; j < insz.d; ++j){

                    size_t ci = i % outsz.w, cr = i / outsz.w;

                    const SN_Base::snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * j;

                    _mm256_storeu_ps(pOut, _mm256_loadu_ps(pIn));
                    _mm256_storeu_ps(pOut + M, _mm256_loadu_ps(pIn + insz.w));
                    _mm256_storeu_ps(pOut + 2 * M, _mm256_loadu_ps(pIn + 2 * insz.w));
                    _mm256_storeu_ps(pOut + 3 * M, _mm256_loadu_ps(pIn + 3 * insz.w));
                    _mm256_storeu_ps(pOut + 4 * M, _mm256_loadu_ps(pIn + 4 * insz.w));
                    _mm256_storeu_ps(pOut + 5 * M, _mm256_loadu_ps(pIn + 5 * insz.w));
                    _mm256_storeu_ps(pOut + 6 * M, _mm256_loadu_ps(pIn + 6 * insz.w));

                    pOut += M * M;
                }
            }
        }

        /////////////////////////////////////////////

        else if (M == 9){

            for (size_t i = 0; i < (outsz.w * outsz.h); ++i){

                for (size_t j = 0; j < insz.d; ++j){

                    size_t ci = i % outsz.w, cr = i / outsz.w;

                    const SN_Base::snFloat* pIn = input + S * insz.w * cr + S * ci + insz.w * insz.h * j;

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
                    pOut += M * M;
                }
            }
        }
    };
    

    template<size_t M>
    void convolutionFWD(size_t S, const snFloat* weight,
        const snSize& insz, const snFloat* input, const snSize& outsz, snFloat* output, snFloat* buff){
               
        const size_t wStepByD = M * M,
                     wStepByK = wStepByD * insz.d,
                     wStepByN = wStepByK * outsz.d,
                     inStepByD = insz.w * insz.h,     
                     inStepByN = inStepByD * insz.d,  
                     outStepByD = outsz.w * outsz.h,  
                     outStepByN = outStepByD * outsz.d;
        
        for (size_t i = 0; i < insz.n; ++i){
                        
            /// Reorder input
            reorderInputCHW2HCW<M>(S, insz, input, outsz, buff);

            cblas_sgemm(CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasTrans,
                blasint(outsz.d),                      // W, rows
                blasint(outsz.w * outsz.h),            // In, cols
                blasint(wStepByK),                     // W, cols, In, rows              
                1.0F,                                  // α
                weight,                                // W
                blasint(wStepByK),                     // W, step to next W
                buff,                                  // In
                blasint(wStepByK),                     // In, step to next In (In21 - In11) 
                0.0,                                   // β
                output,                                // Out
                blasint(outsz.w * outsz.h));           // Out, step to next Out (Y21 - Y11) 

            // +bias on all out layers
            const snFloat* pW = weight + wStepByN;
            for (size_t i = 0; i < outsz.d; ++i){

                snFloat* pOut = output + (outsz.w * outsz.h) * i;
                float bias = *(pW + i);
                for (size_t j = 0; j < (outsz.w * outsz.h); ++j)
                    pOut[j] += bias;
            }

            input += inStepByN;
            output += outStepByN;
        }        
    }

    template <size_t M>
    void defaultFWD(size_t S, size_t D, const snFloat* weight, const snSize& insz, const snFloat* input, const snSize& outsz, snFloat* output){

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

                const snFloat* pIn = input + inStepByN * n,
                    *pW = weight;

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
        const snFloat* weight,
        const snSize& insz, const snFloat* input,
        const snSize& outsz, snFloat* output, snFloat* buff){
        
        if (D == 1){

#define cfwd(MS)   \
    if (M == MS){  \
        convolutionFWD<MS>(S, weight, insz, input, outsz, output, buff); return true; };

            cfwd(1)
            cfwd(3)
            cfwd(5)
            cfwd(7)
            cfwd(9)

            return false;
        }
        else{

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
#undef cfwd
#undef dfwd
    };
}