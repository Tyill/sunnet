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


void Convolution::forwardCPU(size_t kernel, size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output){
   
    size_t wStepByD = fWidth * fHeight,        // шаг весов по входу
           wStepByK = wStepByD * insz.d,       // шаг весов по выходу
           inStepByD = insz.w * insz.h,        // шаг вх слоя по входу
           inStepByN = inStepByD * insz.d,     // шаг вх слоя по батчу
           outStepByD = outsz.w * outsz.h,     // шаг вых слоя по выходу
           outStepByN = outStepByD * outsz.d;  // шаг вых слоя по батчу

    size_t shareStepByN = insz.d + kernel;     // для локализации памяти
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));
    
    memset(output, 0, outStepByN * insz.n * sizeof(snFloat));
        
    // по батчу
#pragma omp parallel for
    for (int n = 0; n < insz.n; ++n){

        snFloat* inBuff = share + shareStepByN * n;
        snFloat* outBuff = share + insz.d + shareStepByN * n;
        
        for (size_t p = 0; p < outStepByD; ++p){
        
            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            memset(outBuff, 0, kernel * sizeof(snFloat));

            // ядро свертки
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pIn = input + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN;
                snFloat* pW = weight + cx + cy * fWidth;

                for (size_t d = 0; d < insz.d; ++d){
                    inBuff[d] = *pIn;
                    pIn += inStepByD;
                }
                            
                // по всем вых слоям
                for (size_t k = 0; k < kernel; ++k){
                                        
                    // по всем вх слоям
                    snFloat cout = 0;
                    for (size_t d = 0; d < insz.d; ++d){
                        cout += inBuff[d] * (*pW);
                        pW += wStepByD;
                    }
                    pW += 1;           // bias;
                    outBuff[k] += cout;
                }
            }

            snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
            snFloat* pW = weight + wStepByK;

            // по всем вых слоям
            for (size_t k = 0; k < kernel; ++k){
               
                *pOut += outBuff[k] + *(pW + k); // + bias
               
                pW += wStepByK;
                pOut += outStepByD;
            }
        }        
    }

    free(share); 
}

void Convolution::backwardCPU_GW(size_t kernel, size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut){
    
    size_t wStepByD = fWidth * fHeight,                  // шаг весов по входу
        wStepByK = wStepByD * insz.d,                 // шаг весов по выходу
        wStepByN = (wStepByK + 1) * kernel,           // шаг весов по батчу
        inStepByD = insz.w * insz.h,                  // шаг вх слоя по входу
        inStepByN = inStepByD * insz.d,               // шаг вх слоя по батчу
        outStepByD = outsz.w * outsz.h,               // шаг вых слоя по выходу
        outStepByN = outStepByD * outsz.d;            // шаг вых слоя по батчу

    size_t shareStepByN = insz.d + kernel + insz.d;      // для локализации памяти
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
                snFloat* pIn = input + (cx + posW + cx * (dilate - 1)) + (cy + posH + cy * (dilate - 1)) * insz.w + n * inStepByN;
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

void Convolution::backwardCPU_G(size_t kernel, size_t fWidth, size_t fHeight, size_t dilate, size_t stride,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut){

    size_t wStepByD = fWidth * fHeight,                  // шаг весов по входу
        wStepByK = wStepByD * insz.d,                 // шаг весов по выходу
        wStepByN = (wStepByK + 1) * kernel,           // шаг весов по батчу
        inStepByD = insz.w * insz.h,                  // шаг вх слоя по входу
        inStepByN = inStepByD * insz.d,               // шаг вх слоя по батчу
        outStepByD = outsz.w * outsz.h,               // шаг вых слоя по выходу
        outStepByN = outStepByD * outsz.d;            // шаг вых слоя по батчу

    size_t shareStepByN = kernel + insz.d;          // для локализации памяти
    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));

    memset(gradOut, 0, inStepByN * insz.n * sizeof(snFloat));

    // по батчу  
#pragma omp parallel for
    for (int n = 0; n < insz.n; ++n){

        snFloat* ginBuff = share + shareStepByN * n;
        snFloat* goutBuff = share + kernel + shareStepByN * n;

        for (size_t p = 0; p < outStepByD; ++p){

            size_t ox = p % outsz.w, oy = p / outsz.w,
                posW = ox * stride, posH = oy * stride;

            snFloat* pGrIn = gradIn + ox + oy * outsz.w + n * outStepByN;

            // по всем вых слоям
            for (size_t k = 0; k < kernel; ++k){
                ginBuff[k] = *pGrIn;
                pGrIn += outStepByD;
            }

            // ядро свертки
            for (size_t c = 0; c < (fWidth * fHeight); ++c){

                size_t cx = c % fWidth, cy = c / fWidth;
                snFloat* pW = weight + cx + cy * fWidth;
                              
                memset(goutBuff, 0, insz.d * sizeof(snFloat));

                // по всем вых слоям
                for (size_t k = 0; k < kernel; ++k){

                    // по всем вх слоям
                    snFloat gin = ginBuff[k];
                    for (size_t d = 0; d < insz.d; ++d){
                        goutBuff[d] += gin * (*pW);
                        pW += wStepByD;
                    }
                    pW += 1;           // bias;
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


#ifndef SN_CUDA

/// иниц вспом параметров CUDA          
void Convolution::iniParamCUDA(const snSize& insz, const snSize& outsz, size_t fWidth, size_t fHeight, map<string, void*>& gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

/// освоб вспом параметров CUDA          
void Convolution::freeParamCUDA(map<string, void*>& gpuPrm){
    ERROR_MESS("CUDA non compiler");
}

void Convolution::forwardCUDA(size_t kernel, size_t fWidth, size_t fHeight, size_t fDilate, size_t stride,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, std::map<std::string, void*>& auxPrm){
    ERROR_MESS("CUDA non compiler");
}

void Convolution::backwardCUDA_GW(size_t kernel, size_t fWidth, size_t fHeight, size_t fDilate, size_t stride,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, map<string, void*>&){
    ERROR_MESS("CUDA non compiler");

}

void Convolution::backwardCUDA_G(size_t kernel, size_t fWidth, size_t fHeight, size_t fDilate, size_t stride,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, map<string, void*>&){
    ERROR_MESS("CUDA non compiler");
}

#endif

#ifndef SN_OpenCL

/// иниц вспом параметров CUDA          
void Convolution::iniParamOCL(const snSize& insz, const snSize& outsz, size_t fWidth, size_t fHeight, map<string, void*>& gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

/// освоб вспом параметров CUDA          
void Convolution::freeParamOCL(map<string, void*>& gpuPrm){
    ERROR_MESS("OpenCL non compiler");
}

void Convolution::forwardOCL(size_t kernel, size_t fWidth, size_t fHeight, size_t fDilate, size_t stride,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* output, std::map<std::string, void*>& auxPrm){
    ERROR_MESS("OpenCL non compiler");
}

void Convolution::backwardOCL_GW(size_t kernel, size_t fWidth, size_t fHeight, size_t fDilate, size_t stride,
    snFloat* weight, const snSize& insz, snFloat* input, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, snFloat* dWeightOut, map<string, void*>&){
    ERROR_MESS("OpenCL non compiler");

}

void Convolution::backwardOCL_G(size_t kernel, size_t fWidth, size_t fHeight, size_t fDilate, size_t stride,
    snFloat* weight, const snSize& insz, const snSize& outsz, snFloat* gradIn, snFloat* gradOut, map<string, void*>&){
    ERROR_MESS("OpenCL non compiler");
}

#endif