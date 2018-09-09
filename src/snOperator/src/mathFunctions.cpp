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

using namespace std;
using namespace SN_Base;


void batchNormForward(snSize insz, snFloat* in, snFloat* out, batchNorm prm){
  
    size_t inSz = insz.w * insz.h * insz.d, bsz = insz.n;

    /// μ = 1/n * ∑x
    cblas_sgemv(CBLAS_ORDER::CblasRowMajor,
        CBLAS_TRANSPOSE::CblasTrans,
        bsz,                          // x, строк - размер батча
        inSz,                         // x, столбцов 
        1.F / bsz,                    // коэф
        in,                           // x, данные
        inSz,                         // x, шаг до след 
        prm.onc,                      // 1й вектор
        1,                            // 1й вектор, шаг движения по вектору
        0.0,                          // коэф
        prm.mean,                     // μ, результ
        1);                           // μ, шаг до след
     
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

void batchNormBackward(snSize insz, snFloat* gradIn, snFloat* gradOut, batchNorm prm){
    // https://kevinzakka.github.io/2016/09/14/batch_normalization/

    size_t inSz = insz.w * insz.h * insz.d, bsz = insz.n;
  
    /// ∂f/∂β = ∑∂f/∂y
    cblas_sgemv(CBLAS_ORDER::CblasRowMajor,
        CBLAS_TRANSPOSE::CblasTrans,
        bsz,                          // ∂f/∂y, строк - размер батча
        inSz,                         // ∂f/∂y, столбцов 
        1.F,                          // коэф
        gradIn,                       // ∂f/∂y, данные
        inSz,                         // ∂f/∂y, шаг до след
        prm.onc,                      // 1й вектор
        1,                            // 1й вектор, шаг движения по вектору
        0.0,                          // коэф
        prm.dSchift,                  // ∂f/∂β, результ
        1);

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