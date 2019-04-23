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
     
   
    template<size_t M, size_t S, size_t D, size_t RO>
    void convolutionFWD(snFloat* weight,
        const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){
     
        /// Reorder input
        buf_t inHCWBuff(snSize(M * M * insz.d, outsz.w, outsz.h), 8);
                
        reorderCHW2HCW<M, S, D, RO>(insz, input, outsz, inHCWBuff.p);

        ///////////////////////////////////

        const size_t wStepByD = M * M,
                     wStepByK = wStepByD * insz.d,
                     wStepByN = wStepByK * outsz.d;
                     
        auto core = std::thread::hardware_concurrency();
        if (core == 0) core = 4;
 
#pragma omp parallel for num_threads(core)
        for (int od = 0; od < int(outsz.d); ++od){

            for (size_t oi = 0; oi < (outsz.w * outsz.h) / RO; ++oi){
                              
                snFloat* pOut = output + (oi * RO) + od * (outsz.w * outsz.h),
                       * pW = weight + wStepByK * od,
                       * pIn = inHCWBuff.p + (oi * RO) * M * M * insz.d;

                snFloat bias = *(weight + wStepByN + od);
                                                
                if (M == 3){

                    CREATE_14REG(arO);
                    CREATE_REG(arW);
                    CREATE_REG(arIn);

                    for (size_t k = 0; k < insz.d; ++k){

                        LOAD_REG(pW, 0, arW);
                                                
                        SUMM_14REG(pIn, M * M, arIn, arW, arO);

                        pIn += M * M * RO;
                        pW += M * M;
                    }
                                      
                    pOut[0] = bias + horSummReg<__m256>(arO0);
                    pOut[1] = bias + horSummReg<__m256>(arO1);
                    pOut[2] = bias + horSummReg<__m256>(arO2);
                    pOut[3] = bias + horSummReg<__m256>(arO3);
                    pOut[4] = bias + horSummReg<__m256>(arO4);
                    pOut[5] = bias + horSummReg<__m256>(arO5);
                    pOut[6] = bias + horSummReg<__m256>(arO6);
                    pOut[7] = bias + horSummReg<__m256>(arO7);
                    pOut[8] = bias + horSummReg<__m256>(arO8);
                    pOut[9] = bias + horSummReg<__m256>(arO9);
                    pOut[10] = bias + horSummReg<__m256>(arO10);
                    pOut[11] = bias + horSummReg<__m256>(arO11);
                    pOut[12] = bias + horSummReg<__m256>(arO12);
                    pOut[13] = bias + horSummReg<__m256>(arO13);

                    pIn = inHCWBuff.p + (oi * RO) * M * M * insz.d;
                    pW = weight + wStepByK * od;

                    getPeakOutput<M, RO>(insz.d, pIn, pW, pOut);
                }                
            }

            if ((outsz.w * outsz.h) % RO){

                for (size_t oi = 0; oi < ((outsz.w * outsz.h) % RO); ++oi){

                    size_t ow = oi % outsz.w, oh = oi / outsz.w;

                    size_t offs = ((outsz.w * outsz.h) / RO) * RO;

                    snFloat* pOut = output + oi + offs + od * (outsz.w * outsz.h),
                           * pW = weight + wStepByK * od,
                           * pIn = inHCWBuff.p + (oi + offs) * M * M * insz.d;

                    snFloat bias = *(weight + wStepByN + od);
                                       
                    if (M == 3){

                        /*CREATE_REG(arO);
                        CREATE_REG(arW);
                        CREATE_REG(arIn);

                        for (size_t k = 0; k < insz.d; ++k){

                            LOAD_REG(pW, 0, arW);
                            
                            SUMM_REG(pIn, 0, arIn, arW, arO);
                                                    
                            pIn += M * M * RO;
                            pW += M * M;
                        }

                        pIn = inHCWBuff.p + (oi + offs) * M * M * insz.d;
                        pW = weight + wStepByK * od;

                        pOut[0] = bias + horSummReg<__m256>(arO);

*/
                   }
                }
            }
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


        
#define cfwd(MS, SS, DS, RO)                  \
    if ((M == MS) && (S == SS) && (D == DS)){  \
        convolutionFWD<MS, SS, DS, RO>(weight, insz, input, outsz, output); return true; };

            cfwd(1, 1, 1, 14)
            cfwd(3, 1, 1, 14)
            cfwd(5, 1, 1, 2)
            cfwd(7, 1, 1, 1)
            cfwd(9, 1, 1, 1)

            cfwd(1, 2, 1, 14)
            cfwd(3, 2, 1, 14)
            cfwd(5, 2, 1, 2)
            cfwd(7, 2, 1, 1)
            cfwd(9, 2, 1, 1)

            /*  cfwd(1, 1, 2)
            cfwd(3, 1, 2)
            cfwd(5, 1, 2)
            cfwd(7, 1, 2)
            cfwd(9, 1, 2)

            cfwd(1, 2, 2)
            cfwd(3, 2, 2)
            cfwd(5, 2, 2)
            cfwd(7, 2, 2)
            cfwd(9, 2, 2)*/
  
            return false;
  
#undef cfwd

    };
};

