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

#include "stdafx.h"
#include "batchNormFunctions.h"

using namespace std;
using namespace SN_Base;


void batchNormForward(bool isLern, const snSize& insz, snFloat* in, snFloat* out, const batchNorm& prm){
  
    size_t sz = insz.w * insz.h * insz.d,
           bsz = insz.n;
    
    if (!isLern){
             
        /// y = ^x * γ + β
        for (size_t j = 0; j < bsz; ++j){
                
            snFloat* cin = in + sz * j,
                   * cout = out + sz * j;      
            for (size_t i = 0; i < sz; ++i)
                cout[i] = (cin[i] - prm.mean[i]) * prm.scale[i] / prm.varce[i] + prm.schift[i];
        }
    }
    else{
        /// varce = sqrt(∑xx - mean^2 + e)
        for (size_t i = 0; i < insz.d; ++i){

            snFloat* cin = in + i * insz.w * insz.h,
                   * cmean = prm.mean + i * insz.w * insz.h,
                   * cvarce = prm.varce + i * insz.w * insz.h;

            for (size_t j = 0; j < (insz.w * insz.h); ++j){

                snFloat sum = 0.F, srq = 0.F;
                for (size_t k = 0; k < bsz; ++k){

                    snFloat v = cin[j + k * sz];

                    sum += v;
                    srq += v * v;
                }
                srq /= bsz;

                snFloat mean = sum / bsz;

                cmean[j] = mean;

                cvarce[j] = sqrt(srq - mean * mean + 0.00001F);
            }
        }

        /// norm = (in - mean) / varce
        /// y = ^x * γ + β
        for (size_t j = 0; j < bsz; ++j){

            snFloat* cin = in + j * sz,
                   * cout = out + j * sz,
                   * norm = prm.norm + j * sz;

            for (size_t i = 0; i < sz; ++i){
                norm[i] = (cin[i] - prm.mean[i]) / prm.varce[i];
                cout[i] = norm[i] * prm.scale[i] + prm.schift[i];
            }
        }
    }
}

void batchNormBackward(const snSize& insz, snFloat* gradIn, snFloat* gradOut, const batchNorm& prm){
    // https://kevinzakka.github.io/2016/09/14/batch_normalization/

    size_t sz = insz.w * insz.h * insz.d,
           bsz = insz.n;
  
    /// ∂f/∂γ = ∑∂f/∂y ⋅ ^x
    for (size_t i = 0; i < insz.d; ++i){

        snFloat* cgr = gradIn + i * insz.w * insz.h,
               * norm = prm.norm + i * insz.w * insz.h,
               * cscale = prm.dScale + i * insz.w * insz.h,
               * cschift = prm.dSchift + i * insz.w * insz.h;
        for (size_t j = 0; j < (insz.w * insz.h); ++j){

            snFloat dScale = 0.F, sum = 0.F;
            for (size_t k = 0; k < bsz; ++k){

                snFloat v = cgr[j + k * sz];

                sum += v;
                dScale += v * norm[j + k * sz];
            }

            cschift[j] = sum;
            cscale[j] = dScale;
        }
    }
       

    /// ∂f/∂x = (m⋅γ⋅∂f/∂y − γ⋅∂f/∂β − ^x⋅γ⋅∂f/∂γ) / m⋅σ2
    for (size_t j = 0; j < bsz; ++j){

        snFloat* igr = gradIn + j * sz, 
               * ogr = gradOut + j * sz, 
               * norm = prm.norm + j * sz;
        for (size_t i = 0; i < sz; ++i)
            ogr[i] = prm.scale[i] * (igr[i] * bsz - prm.dSchift[i] - norm[i] * prm.dScale[i]) / (prm.varce[i] * bsz);
    }
    
    for (size_t i = 0; i < sz; ++i){
        prm.schift[i] -= prm.dSchift[i] * prm.lr;
        prm.scale[i] -= prm.dScale[i] * prm.lr;
    }  
}