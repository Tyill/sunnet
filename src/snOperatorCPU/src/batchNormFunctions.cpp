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
#include "Lib/OpenBLAS/cblas.h"
#include "batchNormFunctions.h"

using namespace std;
using namespace SN_Base;


void channelBatchNorm(bool fwBw, bool isLern, const snSize& insz, snFloat* in, snFloat* out, batchNorm prm){

    /* Select 1 output layer from each image in the batch and normalize */

    size_t stepD = insz.w * insz.h, stepN = stepD * insz.d, bsz = insz.n;

    if (!isLern){

        for (size_t i = 0; i < insz.d; ++i){

            /// y = ^x * γ + β
            for (size_t j = 0; j < bsz; ++j){

                snFloat* cin = in + stepN * j + stepD * i,
                       * cout = out + stepN * j + stepD * i;
                for (size_t k = 0; k < stepD; ++k)
                    cout[k] = (cin[k] - prm.mean[k]) * prm.scale[k] / prm.varce[k] + prm.schift[k];
            }
            prm.offset(stepD);
        }
    }
    else{

        snFloat* share = (snFloat*)calloc(stepD * bsz, sizeof(snFloat));
        snSize sz(insz.w, insz.h, 1, insz.n);

        for (size_t i = 0; i < insz.d; ++i){

            snFloat* pSh = share;
            snFloat* pIn = in + stepD * i;
            for (size_t j = 0; j < bsz; ++j){

                memcpy(pSh, pIn, stepD * sizeof(snFloat));
                pSh += stepD;
                pIn += stepN;
            }

            if (fwBw)
                batchNormForward(sz, share, share, prm);
            else
                batchNormBackward(sz, share, share, prm);

            pSh = share;
            snFloat* pOut = out + stepD * i;
            for (size_t j = 0; j < bsz; ++j){
                memcpy(pOut, pSh, stepD * sizeof(snFloat));
                pSh += stepD;
                pOut += stepN;
            }

            prm.offset(stepD);
            prm.norm += stepD * bsz;
        }
        free(share);
    }
}

void layerBatchNorm(bool fwBw, bool isLern, const snSize& insz, snFloat* in, snFloat* out, const batchNorm& prm){

    if (!isLern){
        size_t sz = insz.w * insz.h * insz.d, bsz = insz.n;

        /// y = ^x * γ + β
        for (size_t j = 0; j < bsz; ++j){

            snFloat* cin = in + j * sz, *cout = out + j * sz;
            for (size_t i = 0; i < sz; ++i)
                cout[i] = (cin[i] - prm.mean[i])  * prm.scale[i] / prm.varce[i] + prm.schift[i];
        }
    }
    else{ // isLerning
        if (fwBw) 
            batchNormForward(insz, in, out, prm);        
        else
            batchNormBackward(insz, in, out, prm);
    }
}

void batchNormForward(const SN_Base::snSize& insz, snFloat* in, snFloat* out, batchNorm prm){
  
    size_t inSz = insz.w * insz.h * insz.d, bsz = insz.n;

    /// μ = 1/n * ∑x
    cblas_sgemv(CBLAS_ORDER::CblasRowMajor,
        CBLAS_TRANSPOSE::CblasTrans,
        blasint(bsz),                 
        blasint(inSz),                
        1.F / bsz,                    
        in,                           
        blasint(inSz),                
        prm.onc,                      
        blasint(1),                   
        0.F,                          
        prm.mean,                     
        blasint(1));                  
     
   /// varce = sqrt(∑xx - mean^2 + e)
   for (size_t i = 0; i < inSz; ++i){

        snFloat* cin = in + i, srq = 0.F;
        for (size_t j = 0; j < bsz; ++j){
            srq += cin[0] * cin[0];
            cin += inSz;
        }
        prm.varce[i] = sqrt(srq / bsz - prm.mean[i] * prm.mean[i] + 0.00001F);
    }
      
    /// norm = (in - mean) / varce
    /// y = ^x * γ + β
    for (size_t j = 0; j < bsz; ++j){

        snFloat* cin = in + j * inSz, *cout = out + j * inSz, *norm = prm.norm + j * inSz;

        for (size_t i = 0; i < inSz; ++i){                        
            norm[i] = (cin[i] - prm.mean[i]) / prm.varce[i];
            cout[i] = norm[i] * prm.scale[i] + prm.schift[i];
        }
    }
   
}

void batchNormBackward(const SN_Base::snSize& insz, snFloat* gradIn, snFloat* gradOut, batchNorm prm){
    // https://kevinzakka.github.io/2016/09/14/batch_normalization/

    size_t inSz = insz.w * insz.h * insz.d, bsz = insz.n;
  
    /// ∂f/∂β = ∑∂f/∂y
    cblas_sgemv(CBLAS_ORDER::CblasRowMajor,
        CBLAS_TRANSPOSE::CblasTrans,
        blasint(bsz),                 
        blasint(inSz),                
        1.F,                          
        gradIn,                       
        blasint(inSz),                
        prm.onc,                      
        blasint(1),                   
        0.F,                          
        prm.dSchift,                  
        blasint(1));

    /// ∂f/∂γ = ∑∂f/∂y ⋅ ^x
    for (size_t i = 0; i < inSz; ++i){

        snFloat* igr = gradIn + i, *norm = prm.norm + i, dScale = 0.F;
        for (size_t j = 0; j < bsz; ++j){

            dScale += igr[0] * norm[0];

            norm += inSz;
            igr += inSz;
        }
        prm.dScale[i] = dScale;
    }

    /// ∂f/∂x = (m⋅γ⋅∂f/∂y − γ⋅∂f/∂β − ^x⋅γ⋅∂f/∂γ) / m⋅σ2
    for (size_t j = 0; j < bsz; ++j){

        snFloat* igr = gradIn + j * inSz, *ogr = gradOut + j * inSz, *norm = prm.norm + j * inSz;
        for (size_t i = 0; i < inSz; ++i)
            ogr[i] = prm.scale[i] * (igr[i] * bsz - prm.dSchift[i] - norm[i] * prm.dScale[i]) / (prm.varce[i] * bsz);
    }
    for (size_t i = 0; i < inSz; ++i){
        prm.schift[i] -= prm.dSchift[i] * prm.lr;
        prm.scale[i] -= prm.dScale[i] * prm.lr;
    }
  
}