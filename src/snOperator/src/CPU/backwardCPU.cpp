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
#include "SNOperator/src/structurs.h"
#include "SNOperator/src/mathFunctions.h"
#include <omp.h>  

using namespace std;
using namespace SN_Base;


#ifdef SN_CPU

void bwdFullyConnected(size_t kernel, snFloat* weight,
    snSize insz, snFloat* input, snFloat* gradIn, snFloat* gradOut, snFloat* dWOut){

    size_t imSz = insz.w * insz.h * insz.d + 1;
    
    // Градиент по весам
    // dW = αIn^T * GrIn + βdW
    // In - матрица вход данных с предыд слоя
    // GrIn - матрица градиентов со след слоя
    cblas_sgemm(CBLAS_ORDER::CblasRowMajor,
        CBLAS_TRANSPOSE::CblasTrans,
        CBLAS_TRANSPOSE::CblasNoTrans,
        imSz,                          // In, строк, кол-во вх значений (+1 - X0)     
        kernel,                        // GrIn, столбцов, кол-во скрытых нейронов 
        insz.n,                        // In, столбцов. GrIn, строк, размер батча                   
        1.0F / insz.n,                 // α коэф 
        input,                         // In, - вх данные - вх значения пришедшие с предыд слоя
        imSz,                          // In, - шаг до след
        gradIn,                        // GrIn - градиент пришедший со след слоя
        kernel,                        // GrIn - шаг до след
        0.0F,                          // β коэф 
        dWOut,                         // dW, выходные данные - градиент по весам
        kernel);                       // dW, шаг до след
        
    // Градиент для предыд слоя
    // GrOut = αGrIn * W^T + βGrOut
    // GrIn - матрица градиентов со след слоя
    // W - веса
    cblas_sgemm(CBLAS_ORDER::CblasRowMajor,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_TRANSPOSE::CblasTrans,
        insz.n,                        // GrIn, строк, размер батча     
        imSz - 1,                      // W, столбцов, кол-во вх значений 
        kernel,                        // GrIn, столбцов. W, строк, кол-во скрытых нейронов                 
        1.0F,                          // α, коэф 
        gradIn,                        // GrIn, градиент пришедший со след слоя
        kernel,                        // GrIn, шаг до след X (X21 - X11) 
        weight + kernel,               // W, веса
        kernel,                        // W, шаг до след W (W21 - W11) 
        0.0F,                          // β, доп коэф 
        gradOut,                       // GrOut, градиент для предыд слоя
        imSz - 1);                     // GrOut, шаг до след Y (Y21 - Y11) 
}

void bwdConvolution(size_t kernel, size_t fWidth, size_t fHeight, size_t stride, 
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){
    
    size_t wStepByD = fWidth * fHeight,                  // шаг весов по входу
           wStepByK = wStepByD * insz.d,                 // шаг весов по выходу
           wStepByN = (wStepByK + 1) * kernel,           // шаг весов по батчу
           inStepByD = insz.w * insz.h,                  // шаг вх слоя по входу
           inStepByN = inStepByD * insz.d,               // шаг вх слоя по батчу
           outStepByD = outsz.w * outsz.h,               // шаг вых слоя по выходу
           outStepByN = outStepByD * outsz.d;            // шаг вых слоя по батчу

    size_t shareStepByN = insz.d + kernel + insz.d;          // для локализации памяти
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    snFloat* wgThr = (insz.n == 1) ? dWeightOut : (snFloat*)calloc(wStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));
    memset(dWeightOut, 0, wStepByN * sizeof(snFloat));
   
     // по батчу  
#pragma omp parallel for
    for (int n = 0; n < insz.n; ++n){
          
        snFloat* inBuff = share + shareStepByN * n;
        snFloat* ginBuff = share + insz.d + shareStepByN * n;
        snFloat* goutBuff = share + insz.d + kernel + shareStepByN * n;
        snFloat* wBuff = wgThr + wStepByN * n;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;
                    
            snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;
            snFloat* pdW = wBuff + wStepByK;

            // по всем вых слоям
            for (size_t k = 0; k < kernel; ++k){                                   
                ginBuff[k] = *pGrIn;             
               
                *(pdW + k) += *pGrIn;      // + bias
            
                pGrIn += outStepByD;
                pdW += wStepByK;
            }
                       
            // ядро свертки
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pIn = input + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;                
                snFloat* pW = weight + cx + cy * fWidth;
                snFloat* pdW = wBuff + cx + cy * fWidth;

                for (size_t d = 0; d < insz.d; ++d){
                    inBuff[d] = *pIn;
                    pIn += inStepByD;
                }

                memset(goutBuff, 0, insz.d * sizeof(snFloat));

                // по всем вых слоям
                for (size_t k = 0; k < kernel; ++k){

                    // по всем вх слоям
                    snFloat gin = ginBuff[k];
                    for (size_t d = 0; d < insz.d; ++d){
                        goutBuff[d] += gin * (*pW);
                        pW += wStepByD;    

                        *pdW += gin * inBuff[d];
                        pdW += wStepByD;
                    }                                       
                    pW += 1;           // bias;
                    pdW += 1;
                }
                               
                snFloat* pGrOut = gradOut + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;

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

void bwdPooling(int type, size_t kernel, snSize outsz, size_t* outputInx, snFloat* gradIn, snSize insz, snFloat* gradOut){

    size_t inStepByD = insz.w * insz.h,           // шаг вх слоя по входу
        inStepByN = inStepByD * insz.d,        // шаг вх слоя по батчу
        outStepByD = outsz.w * outsz.h,        // шаг вых слоя по выходу
        outStepByN = outStepByD * outsz.d,     // шаг вых слоя по батчу
        kernelSz = kernel * kernel;
        
    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));

    if (type == 0){ // max

        // по батчу
#pragma omp parallel for
        for (int n = 0; n < insz.n; ++n){

            for (size_t p = 0; p < outStepByD; ++p){

                size_t ox = p % outsz.w, oy = p / outsz.w,
                    posW = ox * kernel, posH = oy * kernel;

                size_t* pOutInx = outputInx + ox + oy * outsz.w + n * outStepByN;
                snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;
                snFloat* pGrOut = gradOut + n * inStepByN;

                // по всем вх слоям
                for (size_t d = 0; d < insz.d; ++d){

                    size_t c = *pOutInx, cx = c % kernel, cy = c / kernel;
                    pGrOut[(cx + posW) + (cy + posH) * insz.w] = *pGrIn;

                    pGrIn += outStepByD;
                    pOutInx += outStepByD;
                    pGrOut += inStepByD;
                }
            }
        }
    }
    else{ // mean

        size_t shareStepByN = insz.d;                 // для локализации памяти
        snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

        // по батчу
#pragma omp parallel for
        for (int n = 0; n < insz.n; ++n){

            snFloat* outBuff = share + shareStepByN * n;

            for (size_t p = 0; p < outStepByD; ++p){

                size_t ox = p % outsz.w, oy = p / outsz.w,
                    posW = ox * kernel, posH = oy * kernel;

                snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;

                // по всем вых слоям
                for (size_t k = 0; k < outsz.d; ++k){
                    outBuff[k] = *pGrIn;
                    pGrIn += outStepByD;
                }

                // ядро свертки
                for (size_t c = 0; c < kernelSz; ++c){

                    size_t cx = c % kernel, cy = c / kernel;
                    snFloat* pGrOut = gradOut + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;

                    // по всем вх слоям
                    for (size_t d = 0; d < insz.d; ++d){
                        *pGrOut = outBuff[d] / kernelSz;
                        pGrOut += inStepByD;
                    }
                }
            }
        }

        free(share);
    }
}
   

void bwdBatchNorm(snSize insz, snFloat* gradIn, snFloat* gradOut, batchNorm prm){
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
    for (size_t i = 0; i < inSz; ++i){

        snFloat* igr = gradIn + i, *ogr = gradOut + i, *norm = prm.norm + i,
            varce = prm.varce[i] * bsz, scale = prm.scale[i] / varce,
            dSchift = prm.dSchift[i], dScale = prm.dScale[i];
        for (size_t j = 0; j < bsz; ++j){

            *ogr = scale * (*igr * bsz - dSchift - *norm * dScale);

            norm += inSz;
            igr += inSz;
            ogr += inSz;
        }

        prm.schift[i] -= prm.dSchift[i] * prm.lr;
        prm.scale[i] -= prm.dScale[i] * prm.lr;
    }    
}


#endif