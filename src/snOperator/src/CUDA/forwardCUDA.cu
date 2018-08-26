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

#ifdef SN_CUDA

#include "../stdafx.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "SNOperator/src/mathFunctions.h"

using namespace std;
using namespace SN_Base;

bool fwdMalloc(const string& func, size_t sz, snFloat *d_in){

    int sts = cudaMalloc(reinterpret_cast<void **>(&d_in), sz * sizeof(snFloat));
    if (sts != cudaSuccess) {
        fprintf(stderr, (func + " fwdMalloc device memory allocation error: sts " + to_string(sts) + "\n").c_str());
        return false;
    }
        
    return true;
};
#define fcMalloc(sz, d_in) if (!fwdMalloc("fwdFullyConnected", sz, d_in)) return false;  
#define bnMalloc(sz, d_in) if (!fwdMalloc("fwdBatchNorm", sz, d_in)) return false; 

bool fwdFree(const string& func, snFloat *d_in){

    int sts = cudaFree(d_in);
    if (sts != cudaSuccess) {
        fprintf(stderr, (func + " fwdFree memory free error: sts " + to_string(sts) + "\n").c_str());
        return false;
    }
    return true;
}
#define fcFree(d_in) if (!fwdFree("fwdFullyConnected", d_in)) return false; 
#define bnFree(d_in) if (!fwdFree("fwdBatchNorm", d_in)) return false;

bool fwdMallocAndCopyVectorH2D(const string& func, size_t sz, snFloat *d_in, snFloat *h_in){

    int sts = cudaMalloc(reinterpret_cast<void **>(&d_in), sz * sizeof(snFloat));
    if (sts != cudaSuccess) {
        fprintf(stderr, (func + " fwdMallocAndCopyVectorH2D device memory allocation error: sts " + to_string(sts) + "\n").c_str());
        return false;
    }

    /* Initialize the device matrices with the host matrices */
    sts = cublasSetVector(sz, sizeof(snFloat), h_in, 1, d_in, 1);
    if (sts != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, (func + " fwdMallocAndCopyVectorH2D device access error: sts " + to_string(sts) + "\n").c_str());
        return false;
    }
    return true;
};
#define fcMallocAndCopyVectorH2D(sz, d_in, h_in) if (!fwdMallocAndCopyVectorH2D("fwdFullyConnected", sz, d_in, h_in)) return false; 
#define bnMallocAndCopyVectorH2D(sz, d_in, h_in) if (!fwdMallocAndCopyVectorH2D("fwdBatchNorm", sz, d_in, h_in)) return false;

bool fwdMallocAndCopyMatrixH2D(const string& func, size_t rows, size_t cols, snFloat *d_in, snFloat *h_in){

    int sts = cudaMalloc(reinterpret_cast<void **>(&d_in), rows * cols * sizeof(snFloat));
    if (sts != cudaSuccess) {
        fprintf(stderr, (func + " fwdMallocAndCopyMatrixH2D device memory allocation error: sts " + to_string(sts) + "\n").c_str());
        return false;
    }

    /* Initialize the device matrices with the host matrices */
    sts = cublasSetMatrix(rows, cols, sizeof(snFloat), h_in, cols, d_in, cols);
    if (sts != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, (func + " fwdMallocAndCopyMatrixH2D device access error: sts " + to_string(sts) + "\n").c_str());
        return false;
    }
    return true;
};
#define fcMallocAndCopyMatrixH2D(rows, cols, d_in, h_in) if (!fwdMallocAndCopyMatrixH2D("fwdFullyConnected", rows, cols, d_in, h_in)) return false; 
#define bnMallocAndCopyMatrixH2D(rows, cols, d_in, h_in) if (!fwdMallocAndCopyMatrixH2D("fwdBatchNorm", rows, cols, d_in, h_in)) return false;

bool fwdCopyAndFreeVectorD2H(const string& func, size_t sz, snFloat *d_in, snFloat *h_in){

    /* Read the result back */   
    int sts = cublasGetVector(sz, sizeof(snFloat), d_in, 1, h_in, 1);
    if (sts != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, (func + " fwdCopyD2HAndFree device access error: sts " + to_string(sts) + "\n").c_str());
        return false;
    }

    sts = cudaFree(d_in);
    if (sts != cudaSuccess) {
        fprintf(stderr, (func + " fwdCopyD2HAndFree memory free error: sts " + to_string(sts) + "\n").c_str());
        return false;
    }
    return true;
}
#define fcCopyAndFreeVectorD2H(sz, d_in, h_in) if (!fwdCopyAndFreeVectorD2H("fwdFullyConnected", sz, d_in, h_in)) return false; 
#define bnCopyAndFreeVectorD2H(sz, d_in, h_in) if (!fwdCopyAndFreeVectorD2H("fwdBatchNorm", sz, d_in, h_in)) return false; 

bool fwdCopyAndFreeMatrixD2H(const string& func, size_t rows, size_t cols, snFloat *d_in, snFloat *h_in){

    /* Read the result back */
    int sts = cublasGetMatrix(rows, cols, sizeof(snFloat), d_in, cols, h_in, cols);
    if (sts != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, (func + " fwdCopyD2HAndFree device access error: sts " + to_string(sts) + "\n").c_str());
        return false;
    }

    sts = cudaFree(d_in);
    if (sts != cudaSuccess) {
        fprintf(stderr, (func + " fwdCopyD2HAndFree memory free error: sts " + to_string(sts) + "\n").c_str());
        return false;
    }
    return true;
}
#define fcCopyAndFreeMatrixD2H(rows, cols, d_in, h_in) if (!fwdCopyAndFreeMatrixD2H("fwdFullyConnected", rows, cols, d_in, h_in)) return false; 
#define bnCopyAndFreeMatrixD2H(rows, cols, d_in, h_in) if (!fwdCopyAndFreeMatrixD2H("fwdBatchNorm", rows, cols, d_in, h_in)) return false; 


bool fwdFullyConnected(size_t kernel, snSize insz, snFloat* input, snFloat* weight, snFloat* output){
    
    cublasHandle_t cuHandle = nullptr;
    cublasStatus_t sta = cublasCreate(&cuHandle);
    if (cublasCreate(&cuHandle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "fwdFullyConnected CUBLAS initialization error\n");
        return false;
    }

    size_t inSz = insz.w * insz.h * insz.d + 1, bsz = insz.n;

    snFloat *d_in = 0;
    fcMallocAndCopyMatrixH2D(bsz, inSz, d_in, input);

    snFloat *d_w = 0;    
    fcMallocAndCopyMatrixH2D(inSz, kernel, d_w, weight);

    snFloat *d_out = 0;    
    fcMallocAndCopyMatrixH2D(bsz, kernel, d_out, output);
   
    // Out = α * In * W + βC
    // In - матрица вход данных - значения с предыд слоя
    // W - матрица весов
    // Out - матрица выход данных
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasSgemm(cuHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        insz.n,                        // In, строк, кол-во изобр в батче
        kernel,                        // W, столбцов, кол-во скрытых нейронов 
        insz.w * insz.h * insz.d + 1,  // In, столбцов, В М - строк, кол-во вх нейронов - размер одного изображения из батча. (+1 - X0)                   
        &alpha,                        // α, коэф
        input,                         // In, вх данные - нейроны пришедшие с предыд слоя
        insz.w * insz.h * insz.d + 1,  // In, шаг до след X (X21 - X11) 
        weight,                        // W, веса
        kernel,                        // W, шаг до след W (W21 - W11) 
        &beta,                         // β, коэф
        output,                        // Out, выходные данные - нейроны для след слоя
        kernel);                       // Out, шаг до след Y (Y21 - Y11) 
       
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "fwdFullyConnected kernel execution error.\n");
        return false;
    }
        
    fcFree(d_in);

    fcFree(d_w);

    fcCopyAndFreeMatrixD2H(bsz, kernel, d_out, output);

    if (cublasDestroy(cuHandle) != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "fwdFullyConnected shutdown error (A)\n");
}

bool fwdConvolution(size_t kernel, size_t fWidth, size_t fHeight, size_t stride,
    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* output){

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
                snFloat* pIn = input + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;
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

bool fwdPooling(int type, size_t kernel, snSize insz, snFloat* input,
    snSize outsz, snFloat* output, size_t* outputInx){

    size_t inStepByD = insz.w * insz.h,           // шаг вх слоя по входу
           inStepByN = inStepByD * insz.d,        // шаг вх слоя по батчу
           outStepByD = outsz.w * outsz.h,        // шаг вых слоя по выходу
           outStepByN = outStepByD * outsz.d,     // шаг вых слоя по батчу
           kernelSz = kernel * kernel;
   
    size_t* shareI = (size_t*)calloc(insz.d * insz.n, sizeof(size_t));
    snFloat* shareF = (snFloat*)calloc(insz.d * insz.n, sizeof(snFloat));

    memset(output, 0, outStepByN * insz.n * sizeof(snFloat));
    memset(outputInx, 0, outStepByN * insz.n * sizeof(snFloat));

    if (type == 0){ // max

        // по батчу
#pragma omp parallel for
        for (int n = 0; n < insz.n; ++n){

            snFloat* outBuff = shareF + insz.d * n;
            size_t* outInxBuff = shareI + insz.d * n;

            for (size_t p = 0; p < outStepByD; ++p){

                size_t ox = p % outsz.w, oy = p / outsz.w,
                    posW = ox * kernel, posH = oy * kernel;

                memset(outBuff, 0, insz.d * sizeof(snFloat));
                memset(outInxBuff, 0, insz.d * sizeof(size_t));

                // ядро свертки
                for (size_t c = 0; c < kernelSz; ++c){

                    size_t cx = c % kernel, cy = c / kernel;
                    snFloat* pIn = input + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;

                    // по всем вх слоям
                    for (size_t d = 0; d < insz.d; ++d){
                        snFloat val = *pIn;
                        pIn += inStepByD;
                        if (val > outBuff[d]){
                            outBuff[d] = val;
                            outInxBuff[d] = c;
                        }
                    }
                }

                snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
                size_t* pOutInx = outputInx + ox + oy * outsz.w + n * outStepByN;

                // по всем вых слоям
                for (size_t k = 0; k < outsz.d; ++k){

                    *pOut = outBuff[k];
                    *pOutInx = outInxBuff[k];

                    pOut += outStepByD;
                    pOutInx += outStepByD;
                }
            }
        }
    }
    else{ // mean

        // по батчу
#pragma omp parallel for
        for (int n = 0; n < insz.n; ++n){

            snFloat* outBuff = shareF + insz.d * n;
          
            for (size_t p = 0; p < outStepByD; ++p){

                size_t ox = p % outsz.w, oy = p / outsz.w,
                    posW = ox * kernel, posH = oy * kernel;

                memset(outBuff, 0, insz.d * sizeof(snFloat));
              
                // ядро свертки
                for (size_t c = 0; c < kernelSz; ++c){

                    size_t cx = c % kernel, cy = c / kernel;
                    snFloat* pIn = input + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;

                    // по всем вх слоям
                    for (size_t d = 0; d < insz.d; ++d){
                        outBuff[d] += *pIn;
                        pIn += inStepByD;
                    }
                }

                snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;

                // по всем вых слоям
                for (size_t k = 0; k < outsz.d; ++k){
                    *pOut = outBuff[k] / kernelSz;
                    pOut += outStepByD;
                }
            }
        }
    }
   
    free(shareI); 
    free(shareF);
}

bool fwdBatchNorm(snSize insz, snFloat* in, snFloat* out, batchNorm prm){
     
    size_t inSz = insz.w * insz.h * insz.d, bsz = insz.n;

    cublasHandle_t cuHandle;
    if (cublasCreate(&cuHandle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "fwdFullyConnected CUBLAS initialization error\n");
        return;
    }

    snFloat *d_in = 0;
    bnMallocAndCopyMatrixH2D(bsz, inSz, d_in, in);

    snFloat *d_onc = 0;
    bnMallocAndCopyVectorH2D(bsz, d_onc, prm.onc);

    snFloat *d_mean = 0;
    bnMallocAndCopyVectorH2D(inSz, d_mean, prm.mean);

    /// μ = 1/n * ∑x
    float alpha = 1.0f / bsz;
    float beta = 0.0f;
    cublasStatus_t status = cublasSgemv(cuHandle,
        CUBLAS_OP_T,
        bsz,                          // x, строк - размер батча
        inSz,                         // x, столбцов 
        &alpha,                       // коэф
        d_in,                         // x, данные
        inSz,                         // x, шаг до след 
        d_onc,                        // 1й вектор
        1,                            // 1й вектор, шаг движения по вектору
        &beta,                        // коэф
        d_mean,                       // μ, результ
        1);                           // μ, шаг до след
           
   

    bnCopyD2HAndFree(inSz, d_mean, prm.mean);
   
    bnFree(d_in);

    bnFree(d_onc);

    if (cublasDestroy(cuHandle) != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "fwdFullyConnected shutdown error (A)\n");

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

#undef fcMalloc
#undef fcMallocAndCopyH2D
#undef fcCopyD2HAndFree
#undef fcFree

#undef bnMalloc
#undef bnMallocAndCopyH2D
#undef bnCopyD2HAndFree
#undef bnFree

#endif //#ifdef SN_CPU