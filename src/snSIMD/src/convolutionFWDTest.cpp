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

#include "snBase/snBase.h"
#include "Lib/OpenBLAS/cblas.h"

using namespace std;
using namespace SN_Base;

namespace SN_SIMD{

    struct buf_t{

        SN_Base::snFloat* p = nullptr;
        SN_Base::snSize sz;

        buf_t(const SN_Base::snSize& size = SN_Base::snSize(0, 0, 0, 0, 0), size_t add = 0) : sz(size) {

            if (size.size() > 0)
                //p = (SN_Base::snFloat*)_mm_malloc((size.size() + add) * sizeof(SN_Base::snFloat), 64);
                p = (SN_Base::snFloat*)malloc((size.size() + add) * sizeof(SN_Base::snFloat));
        }

        ~buf_t() {
            if (p) free(p);
        }

    };
   
    template<size_t M, size_t S, size_t D>
    void reorderInputCHW2HCW(const SN_Base::snSize& insz, const SN_Base::snFloat* input, const SN_Base::snSize& outsz, SN_Base::snFloat* output){

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

        else if ((M == 3) && (D == 1)){

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

        else if ((M == 5) && (D == 1)){

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

        else if ((M == 7) && (D == 1)){

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

        else if ((M == 9) && (D == 1)){

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

        ///////////////////////////////////////////////////
        // TODO

        /*else if ((M == 3) && (D == 2)){

        }

        else if ((M == 5) && (D == 2)){

        }

        else if ((M == 7) && (D == 2)){

        }

        else if ((M == 9) && (D == 2)){


        }*/
    };

    template<size_t M, size_t S, size_t D>
    void convolutionFWD(const snFloat* weight,
        const snSize& insz, const snFloat* input, const snSize& outsz, snFloat* output){
     
        /// Reorder input
        buf_t inHCWBuff(snSize(M * M * insz.d, outsz.w, outsz.h), 8);
                
        reorderInputCHW2HCW<M, S, D>(insz, input, outsz, inHCWBuff.p);

      
        ///////////////////////////////////

        const size_t wStepByD = M * M,
                     wStepByK = wStepByD * insz.d,
                     wStepByN = wStepByK * outsz.d,                   
                     imSz = inHCWBuff.sz.size();
        
        // Out = α * In * W + βC
        // In - data input matrix - values from the previous layer
        // W - weights matrix
        // Out - data output matrix
        cblas_sgemm(CBLAS_ORDER::CblasRowMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    blasint(outsz.d),                      // In, rows
                    blasint(outsz.w * outsz.h),            // W, cols
                    blasint(M * M * insz.d),               // In, cols, W, rows              
                    1.0F,                                  // α
                    inHCWBuff.p,                           // In
                    blasint(M * M * insz.d),               // In, step to next In
                    weight,                                // W
                    blasint(outsz.w * outsz.h),                      // W, step to next W (W21 - W11) 
                    0.0,                                   // β
                    output,                                // Out
                    blasint(outsz.w * outsz.h));                     // Out, step to next Out (Y21 - Y11) 

        /*auto core = std::thread::hardware_concurrency();
        if (core == 0) core = 4;
        
#pragma omp parallel for num_threads(core)
        for (int od = 0; od < int(outsz.d); ++od){
             
            const snFloat bias = *(weight + wStepByN + od);
                     
            for (size_t oi = 0; oi < (outsz.w * outsz.h) / RO; ++oi){

                const snFloat* pW = weight + wStepByK * od,
                             * pIn = inHCWBuff.p + (oi * RO) * M * M * insz.d;

                snFloat* pOut = output + (oi * RO) + od * (outsz.w * outsz.h);
                                           
                kernel<M, RO>(core, pW, bias, insz, pIn, outsz, pOut);
            }
                             
            if (peak){
                
                const size_t offs = ((outsz.w * outsz.h) / RO) * RO;
                
                const snFloat* pW = weight + wStepByK * od,
                             * pIn = inHCWBuff.p + offs * M * M * insz.d;
                        
                snFloat* pOut = output + offs + od * (outsz.w * outsz.h);
                            
                kernelPeak<M>(peak, pW, bias, insz, pIn, outsz, pOut);
            }                      
        }    */  
    }

//    template <size_t M>
//    void defaultFWD(size_t S, size_t D, const snFloat* weight, const snSize& insz, const snFloat* input, const snSize& outsz, snFloat* output){
//
//        const size_t wStepByD = M * M,          // step weight by input
//            kernel = outsz.d,
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
//        auto core = std::thread::hardware_concurrency();
//        if (core == 0) core = 4;
//
//        // by batch
//#pragma omp parallel for num_threads(core)
//        for (int n = 0; n < int(insz.n); ++n){
//
//            snFloat* outBuff = share + shareStepByN * n;
//            snFloat In[wStepByD], W[wStepByD];
//
//            for (size_t p = 0; p < outStepByD; ++p){
//
//                size_t ox = p % outsz.w, oy = p / outsz.w,
//                    posW = ox * S, posH = oy * S;
//
//                memset(outBuff, 0, kernel * sizeof(snFloat));
//
//                const snFloat* pIn = input + inStepByN * n,
//                             * pW = weight;
//
//                // on all in layers
//                for (size_t d = 0; d < insz.d; ++d){
//
//                    for (size_t c = 0; c < wStepByD; ++c){
//
//                        size_t cx = c % M, cy = c / M;
//                        In[c] = *(pIn + (cx + posW + cx * (D - 1)) + (cy + posH + cy * (D - 1)) * insz.w);
//                    }
//
//                    pW = weight + wStepByD * d;
//
//                    // on all out layers
//                    for (size_t k = 0; k < kernel; ++k){
//
//                        for (size_t c = 0; c < wStepByD; ++c){
//
//                            size_t cx = c % M, cy = c / M;
//                            W[c] = *(pW + cx + cy * M);
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
//                        outBuff[k] += horSummReg<__m256>(arOut);
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
//    

    bool convolutionFWD(size_t M, size_t S, size_t D,
        const snFloat* weight,
        const snSize& insz, const snFloat* input,
        const snSize& outsz, snFloat* output){

      
        /*if ((insz.n > 1) || (S > 2) || (D > 1)){
  
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
#undef dfwd*/


        
#define cfwd(MS, SS, DS)                  \
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
                       
            return false;
  
#undef cfwd

    };
};

