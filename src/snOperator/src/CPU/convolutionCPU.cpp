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

#include "../stdafx.h"
#include "Lib/OpenBLAS/cblas.h"
#include "snOperator/src/Operator/convolution.h"
#include <omp.h>


using namespace std;
using namespace SN_Base;

#ifdef dSN_MKLML

#include "Lib/mklml/mkldnn.hpp"
#include <numeric>

namespace mkl = mkldnn;

struct mklParams{





};

void Convolution::iniParamCPU(bool isLern, const SN_Base::snSize& insz, const SN_Base::snSize& outsz,
    const convParams&, void** cpuPrm){

    bool isFirst = false;

    mklParams* mklPrm = (mklParams*)*cpuPrm;
    if (!mklPrm){

        mklPrm = new mklParams();

        using tag = mkl::memory::format_tag;
        using dt = mkl::memory::data_type;

        mkl::engine eng(mkl::engine::cpu, 0);
        mkl::stream s(eng);

        std::vector<mkl::primitive> net;
        std::vector<std::unordered_map<int, mkl::memory>> net_args;

        const mkl::memory::dim batch = 1;

        mkl::memory::dims conv1_src_tz = { batch, 3, 227, 227 };
        mkl::memory::dims conv1_weights_tz = { 96, 3, 11, 11 };
        mkl::memory::dims conv1_bias_tz = { 96 };
        mkl::memory::dims conv1_dst_tz = { batch, 96, 55, 55 };
        mkl::memory::dims conv1_strides = { 4, 4 };
        mkl::memory::dims conv1_padding = { 0, 0 };

        /* Allocate input and output buffers for user data */
        std::vector<float> user_src(batch * 3 * 227 * 227);
        std::vector<float> user_dst(batch * 1000);

        /* Allocate and fill buffers for weights and bias */
        std::vector<float> conv1_weights;//(product(conv1_weights_tz));
        std::vector<float> conv1_bias;//(product(conv1_bias_tz));

        /* create memory for user data */
        auto user_src_memory = mkl::memory({ { conv1_src_tz }, dt::f32, tag::nchw }, eng, user_src.data());
        auto user_weights_memory = mkl::memory({ { conv1_weights_tz }, dt::f32, tag::oihw }, eng, conv1_weights.data());
        auto conv1_user_bias_memory = mkl::memory({ { conv1_bias_tz }, dt::f32, tag::x }, eng, conv1_bias.data());

        /* create memory descriptors for convolution data w/ no specified format */
        auto conv1_src_md = mkl::memory::desc({ conv1_src_tz }, dt::f32, tag::nchw);
        auto conv1_bias_md = mkl::memory::desc({ conv1_bias_tz }, dt::f32, tag::nchw);
        auto conv1_weights_md = mkl::memory::desc({ conv1_weights_tz }, dt::f32, tag::nchw);
        auto conv1_dst_md = mkl::memory::desc({ conv1_dst_tz }, dt::f32, tag::nchw);

        /* create a convolution */
        auto conv1_desc = mkl::convolution_forward::desc(
            mkl::prop_kind::forward_inference,
            mkl::convolution_direct, conv1_src_md, conv1_weights_md, conv1_bias_md,
            conv1_dst_md, conv1_strides, conv1_padding, conv1_padding,
            mkl::padding_kind::zero);

        auto conv1_prim_desc = mkl::convolution_forward::primitive_desc(conv1_desc, eng);

        auto conv1_dst_memory = mkl::memory(conv1_prim_desc.dst_desc(), eng);

        /* create convolution primitive and add it to net */
        net.push_back(mkl::convolution_forward(conv1_prim_desc));
        net_args.push_back({ { MKLDNN_ARG_SRC, user_src_memory },
        { MKLDNN_ARG_WEIGHTS, user_weights_memory },
        { MKLDNN_ARG_BIAS, conv1_user_bias_memory },
        { MKLDNN_ARG_DST, conv1_dst_memory } });
    }

}

void Convolution::freeParamCPU(void* cpuPrm){


}

void Convolution::forwardCPU(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, void* cpuPrm){
     
  

}

void Convolution::backwardCPU_GW(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, void* cpuPrm){



}

void Convolution::backwardCPU_G(const convParams& prms,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, void* cpuPrm){

}

#else

#ifdef SN_AVX
#include <immintrin.h>


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

template <int R>
void forwardAVX(size_t kernel, size_t stride, size_t dilate,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

    const size_t wStepByD = R * R,          // step weight by input
        wStepByK = wStepByD * insz.d,       // step weight by output
        wStepByN = wStepByK * kernel,       // step weight by batch
        inStepByD = insz.w * insz.h,        // step in by input
        inStepByN = inStepByD * insz.d,     // step in by batch
        outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch

    size_t shareStepByN = kernel;           // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));
      
    // by batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* outBuff = share + shareStepByN * n;
        snFloat In[wStepByD], W[wStepByD];

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            memset(outBuff, 0, kernel * sizeof(snFloat));
                       
            snFloat* pIn = input + inStepByN * n;
            snFloat* pW = weight;

            // on all in layers
            for (size_t d = 0; d < insz.d; ++d){
                
                for (size_t c = 0; c < wStepByD; ++c){

                    size_t cx = c % R, cy = c / R;
                    In[c] = *(pIn + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w);
                }

                pW = weight + wStepByD * d;

                // on all out layers
                for (size_t k = 0; k < kernel; ++k){
                                            
                    for (size_t c = 0; c < wStepByD; ++c){

                        size_t cx = c % R, cy = c / R;
                        W[c] = *(pW + cx + cy * R);
                    }

                    __m256 arOut = _mm256_setzero_ps();

                    for (int z = 0; z < wStepByD / 8; ++z){                      
                                           
                        __m256 arIn = _mm256_loadu_ps(In + z * 8);

                        __m256 arW = _mm256_loadu_ps(W + z * 8);

                        arOut = _mm256_add_ps(arOut, _mm256_mul_ps(arIn, arW));
                    }
                    
                    outBuff[k] += horSummAVX(arOut);
 
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

template <int R>
void backwardGW_AVX(size_t kernel, size_t stride, size_t dilate,
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
void backwardG_AVX(size_t kernel, size_t stride, size_t dilate,
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


#endif // SN_AVX

void Convolution::iniParamCPU(bool isLern, const SN_Base::snSize& insz, const SN_Base::snSize& outsz,
    const convParams&, void** cpuPrm){


}

void Convolution::freeParamCPU(void* cpuPrm){


}

void forwardBASE(size_t kernel, size_t fWidth, size_t fHeight, size_t stride, size_t dilate,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){

    size_t wStepByD = fWidth * fHeight,        // step weight by input
           wStepByK = wStepByD * insz.d,       // step weight by output
           wStepByN = wStepByK * kernel,       // step weight by batch
           inStepByD = insz.w * insz.h,        // step in by input
           inStepByN = inStepByD * insz.d,     // step in by batch
           outStepByD = outsz.w * outsz.h,     // step out by input
           outStepByN = outStepByD * outsz.d;  // step out by batch

    size_t shareStepByN = insz.d + kernel;     // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));
     
    // by batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* outBuff = share + insz.d + shareStepByN * n;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            memset(outBuff, 0, kernel * sizeof(snFloat));

            // kernel conv
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pIn = input + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN;
                snFloat* pW = weight + cx + cy * fWidth;

                for (size_t d = 0; d < insz.d; ++d){
                    inBuff[d] = *pIn;
                    pIn += inStepByD;
                }

                // on all out layers
                for (size_t k = 0; k < kernel; ++k){

                    // on all in layers
                    snFloat cout = 0;
                    for (size_t d = 0; d < insz.d; ++d){
                        cout += inBuff[d] * (*pW);
                        pW += wStepByD;
                    }
                    outBuff[k] += cout;
                }
            }

            snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
            snFloat* pW = weight + wStepByN;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){

                *pOut = outBuff[k] + *(pW + k); // + bias              

                pOut += outStepByD;
            }
        }
    }

    free(share);
}

void Convolution::forwardCPU(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, void* cpuPrm){
     
#ifdef SN_AVX
   
    if ((prms.fWidth == 3) && (prms.fHeight == 3))
        forwardAVX<3>(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);
    else if ((prms.fWidth == 5) && (prms.fHeight == 5))
        forwardAVX<5>(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);
    else if ((prms.fWidth == 7) && (prms.fHeight == 7))
        forwardAVX<7>(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);
    else if ((prms.fWidth == 9) && (prms.fHeight == 9))
        forwardAVX<9>(prms.kernel, prms.stride, prms.dilate, weight, insz, input, outsz, output);
    else
        forwardBASE(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate, weight, insz, input, outsz, output);
 
#else

    forwardBASE(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate, weight, insz, input, outsz, output);

#endif
}

void backwardGW_BASE(size_t kernel, size_t fWidth, size_t fHeight, size_t stride, size_t dilate,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){

    size_t wStepByD = fWidth * fHeight,           // step weight by input
           wStepByK = wStepByD * insz.d,          // step weight by output
           wStepByN = wStepByK * kernel + kernel, // step weight by batch
           inStepByD = insz.w * insz.h,           // step in by input
           inStepByN = inStepByD * insz.d,        // step in by batch
           outStepByD = outsz.w * outsz.h,        // step out by input
           outStepByN = outStepByD * outsz.d;     // step out by batch

    size_t shareStepByN = insz.d + kernel + insz.d;      // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    snFloat* wgThr = (insz.n == 1) ? dWeightOut : (snFloat*)calloc(wStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));
    memset(dWeightOut, 0, wStepByN * sizeof(snFloat));

    // by batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* ginBuff = share + insz.d + shareStepByN * n;
        snFloat* goutBuff = share + insz.d + kernel + shareStepByN * n;
        snFloat* wBuff = wgThr + wStepByN * n;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;
            snFloat* pdW = wBuff + wStepByK * kernel;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){
                ginBuff[k] = *pGrIn;
                *(pdW + k) += *pGrIn;      // + bias

                pGrIn += outStepByD;
            }

            // kernel conv
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pIn = input + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN;
                snFloat* pW = weight + cx + cy * fWidth;
                snFloat* pdW = wBuff + cx + cy * fWidth;

                for (size_t d = 0; d < insz.d; ++d){
                    inBuff[d] = *pIn;
                    pIn += inStepByD;
                }
          
                memset(goutBuff, 0, insz.d * sizeof(snFloat));

                // on all out layers
                for (size_t k = 0; k < kernel; ++k){

                    // on all in layers
                    snFloat gin = ginBuff[k];
                    for (size_t d = 0; d < insz.d; ++d){
                        goutBuff[d] += gin * (*pW);
                        pW += wStepByD;

                        *pdW += gin * inBuff[d];
                        pdW += wStepByD;
                    }
                }

                snFloat* pGrOut = gradOut + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN;

                for (size_t d = 0; d < insz.d; ++d){
                    *pGrOut += goutBuff[d];
                    pGrOut += inStepByD;
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

    free(share);

}

void Convolution::backwardCPU_GW(const convParams& prms,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, void* cpuPrm){
    
#ifdef SN_AVX

    if ((prms.fWidth == 3) && (prms.fHeight == 3))
        backwardGW_AVX<3>(prms.kernel, prms.stride, prms.dilate,
           weight, insz, input, outsz, gradIn, gradOut, dWeightOut);
    else if ((prms.fWidth == 5) && (prms.fHeight == 5))
        backwardGW_AVX<5>(prms.kernel, prms.stride, prms.dilate,
           weight, insz, input, outsz, gradIn, gradOut, dWeightOut);
    else if ((prms.fWidth == 7) && (prms.fHeight == 7))
        backwardGW_AVX<7>(prms.kernel, prms.stride, prms.dilate,
           weight, insz, input, outsz, gradIn, gradOut, dWeightOut);
    else if ((prms.fWidth == 9) && (prms.fHeight == 9))
        backwardGW_AVX<9>(prms.kernel, prms.stride, prms.dilate,
           weight, insz, input, outsz, gradIn, gradOut, dWeightOut);
    else
        backwardGW_BASE(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate,
           weight, insz, input, outsz, gradIn, gradOut, dWeightOut);

#else

    backwardGW_BASE(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate,
          weight, insz, input, outsz, gradIn, gradOut, dWeightOut);
#endif
}


void backwardG_Base(size_t kernel, size_t fWidth, size_t fHeight, size_t stride, size_t dilate,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut){

    size_t wStepByD = fWidth * fHeight,         // step weight by input         
           inStepByD = insz.w * insz.h,         // step in by input
           inStepByN = inStepByD * insz.d,      // step in by batch
           outStepByD = outsz.w * outsz.h,      // step out by input
           outStepByN = outStepByD * outsz.d;   // step out by batch

    size_t shareStepByN = kernel + insz.d;          // for local mem
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));

    // by batch
#pragma omp parallel for
    for (int n = 0; n < int(insz.n); ++n){

        snFloat* ginBuff = share + shareStepByN * n;
        snFloat* goutBuff = share + kernel + shareStepByN * n;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;

            // on all out layers
            for (size_t k = 0; k < kernel; ++k){
                ginBuff[k] = *pGrIn;
                pGrIn += outStepByD;
            }

            // kernel conv
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pW = weight + cx + cy * fWidth;
                              
                memset(goutBuff, 0, insz.d * sizeof(snFloat));

                // on all out layers
                for (size_t k = 0; k < kernel; ++k){

                    // on all in layers
                    snFloat gin = ginBuff[k];
                    for (size_t d = 0; d < insz.d; ++d){
                        goutBuff[d] += gin * (*pW);
                        pW += wStepByD;
                    }
                }

                snFloat* pGrOut = gradOut + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN;

                for (size_t d = 0; d < insz.d; ++d){
                    *pGrOut += goutBuff[d];
                    pGrOut += inStepByD;
                }
            }
        }
    }

    free(share);
}

void Convolution::backwardCPU_G(const convParams& prms,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, void* cpuPrm){

#ifdef SN_AVX

    if ((prms.fWidth == 3) && (prms.fHeight == 3))
        backwardG_AVX<3>(prms.kernel, prms.stride, prms.dilate,
           weight, insz, outsz, gradIn, gradOut);
    else if ((prms.fWidth == 5) && (prms.fHeight == 5))
        backwardG_AVX<5>(prms.kernel, prms.stride, prms.dilate,
           weight, insz, outsz, gradIn, gradOut);
    else if ((prms.fWidth == 7) && (prms.fHeight == 7))
        backwardG_AVX<7>(prms.kernel, prms.stride, prms.dilate,
           weight, insz, outsz, gradIn, gradOut);
    else if ((prms.fWidth == 9) && (prms.fHeight == 9))
        backwardG_AVX<9>(prms.kernel, prms.stride, prms.dilate,
           weight, insz, outsz, gradIn, gradOut);
    else
        backwardG_Base(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate,
           weight, insz, outsz, gradIn, gradOut);

#else

    backwardG_Base(prms.kernel, prms.fWidth, prms.fHeight, prms.stride, prms.dilate,
        weight, insz, outsz, gradIn, gradOut);
#endif
}

#endif // SN_MKLML


#ifndef SN_CUDA

/// init aux params CUDA          
void Convolution::iniParamCUDA(bool isLern, const SN_Base::snSize& insz, const SN_Base::snSize& outsz,
    const convParams&, void** gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

/// free aux params CUDA          
void Convolution::freeParamCUDA(void* gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

void Convolution::forwardCUDA(const convParams&,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, void* gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

void Convolution::backwardCUDA_GW(const convParams&,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, void* gpuPrm){
    ERROR_MESS("CUDA non compiler");

}

void Convolution::backwardCUDA_G(const convParams&,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, void* gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

#endif

#ifndef SN_OpenCL

/// init aux params OpenCL          
void Convolution::iniParamOCL(bool isLern, const snSize& insz, const snSize& outsz,
    const convParams&, void** gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

/// free aux params OpenCL           
void Convolution::freeParamOCL(void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void Convolution::forwardOCL(const convParams&,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void Convolution::backwardOCL_GW(const convParams&,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");

}

void Convolution::backwardOCL_G(const convParams&,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, void* gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

#endif