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
#include "snOperator/src/mathFunctions.h"
#include "snOperator/src/random.h"

using namespace std;
using namespace SN_Base;

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

void dropOut(bool isLern, SN_Base::snFloat dropOut, const SN_Base::snSize& outsz, SN_Base::snFloat* out){
        
    if (isLern){
        size_t sz = size_t(outsz.size() * dropOut);
        vector<int> rnd(sz);
        rnd_uniformInt(rnd.data(), sz, 0, int(outsz.size()));

        for (auto i : rnd) out[i] = 0;
    }
    else{
        size_t sz = outsz.size();
        for (size_t i = 0; i < sz; ++i)
            out[i] *= (1.F - dropOut);
    }
}

void paddingOffs(bool in2out, size_t paddW, size_t paddH, const snSize& insz, snFloat* in, snFloat* out){

    /// copy with offset padding for each image    
    size_t sz = insz.h * insz.d * insz.n, stW = insz.w, stH = insz.h;
    if (in2out){
        in += (stW + paddW * 2) * paddH;
        for (size_t i = 0; i < sz; ++i){

            if ((i % stH == 0) && (i > 0))
                in += (stW + paddW * 2) * paddH * 2;

            in += paddW;
            for (size_t j = 0; j < stW; ++j)
                out[j] = in[j];
            in += paddW + stW;

            out += stW;
        }
    }
    else{
        in += (stW + paddW * 2) * paddH;
        for (size_t i = 0; i < sz; ++i){

            if ((i % stH == 0) && (i > 0))
                in += (stW + paddW * 2) * paddH * 2;

            in += paddW;
            for (size_t j = 0; j < stW; ++j)
                in[j] = out[j];
            in += paddW + stW;

            out += stW;
        }
    }
}
