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

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../stdafx.h"
#include "SNOperator/src/mathFunctions.h"
#include "SNOperator/src/Operator/fullyConnected.h"
//#include <omp.h>

using namespace std;
using namespace SN_Base;
          
void FullyConnected::iniParamCUDA(SN_Base::snSize insz, size_t kernel, std::map<std::string, void*>& auxPrm){
    
    size_t ida = insz.w * insz.h * insz.d + 1, bsz = insz.n;

    if (auxPrm.find("cuHandleFWD") == auxPrm.end()){

        cublasHandle_t cuHandle = nullptr;
        int sts = cublasCreate(&cuHandle);
        if (sts != CUBLAS_STATUS_SUCCESS){
            ERROR_MESS("fwdFullyConnected CUBLAS initialization error: sts " + to_string(sts));
            return;
        }
        auxPrm["cuHandleFWD"] = cuHandle;

        snFloat* d_in = 0, *d_w = 0, *d_out = 0;
        cudaMalloc(reinterpret_cast<void**>(&d_in), bsz * ida * sizeof(snFloat)); auxPrm["d_in_FWD"] = d_in;
        cudaMalloc(reinterpret_cast<void**>(&d_w), ida * kernel * sizeof(snFloat)); auxPrm["d_w_FWD"] = d_w;
        cudaMalloc(reinterpret_cast<void**>(&d_out), bsz * kernel * sizeof(snFloat)); auxPrm["d_out_FWD"] = d_out;
    }
    else{
        snFloat* d_in = (snFloat*)auxPrm["d_in_FWD"], 
               * d_w = (snFloat*)auxPrm["d_w_FWD"],
               * d_out = (snFloat*)auxPrm["d_out_FWD"];
        cudaFree(d_in);  cudaMalloc(reinterpret_cast<void**>(&d_in), bsz * ida * sizeof(snFloat)); auxPrm["d_in_FWD"] = d_in;
        cudaFree(d_w);   cudaMalloc(reinterpret_cast<void**>(&d_w), ida * kernel * sizeof(snFloat)); auxPrm["d_w_FWD"] = d_w;
        cudaFree(d_out); cudaMalloc(reinterpret_cast<void**>(&d_out), bsz * kernel * sizeof(snFloat)); auxPrm["d_out_FWD"] = d_out;
    }
}
         
void FullyConnected::freeParamCUDA(std::map<std::string, void*>& auxPrm){
    
    if (auxPrm.find("cuHandleFWD") == auxPrm.end()) return;
      
    cublasDestroy((cublasHandle_t)auxPrm["cuHandleFWD"]);

    snFloat* d_in = (snFloat*)auxPrm["d_in_FWD"],
        *d_w = (snFloat*)auxPrm["d_w_FWD"],
        *d_out = (snFloat*)auxPrm["d_out_FWD"];

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_out);      
}

void FullyConnected::forwardCUDA(size_t kernel, snSize insz, snFloat* input, snFloat* weight, snFloat* output, std::map<std::string, void*>& auxPrm){
      
    if (auxPrm.find("cuHandleFWD") == auxPrm.end()) return;

    size_t ida = insz.w * insz.h * insz.d + 1, bsz = insz.n;
   
    cublasHandle_t cuHandle = (cublasHandle_t)auxPrm["cuHandleFWD"];

    snFloat *d_in = (snFloat*)auxPrm["d_in_FWD"],
            *d_w = (snFloat*)auxPrm["d_w_FWD"], 
            *d_out = (snFloat*)auxPrm["d_out_FWD"];
   
    cublasSetMatrix(bsz, ida, sizeof(snFloat), input, bsz, d_in, bsz);
    
    cublasSetMatrix(ida, kernel, sizeof(snFloat), weight, ida, d_w, ida);
   
    // Out = α * W * In + βC
    // In - матрица вход данных - значения с предыд слоя
    // W - матрица весов
    // Out - матрица выход данных
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cuHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        kernel,                        // W, столбцов, кол-во скрытых нейронов 
        bsz,                           // In, строк, кол-во изобр в батче
        ida,                           // In, столбцов, В М - строк, кол-во вх нейронов - размер одного изображения из батча. (+1 - X0)                   
        &alpha,                        // α, коэф
        d_w,                           // W, веса
        kernel,                        // W, шаг до след W (W21 - W11)
        d_in,                          // In, вх данные - нейроны пришедшие с предыд слоя
        ida,                           // In, шаг до след X (X21 - X11)  
        &beta,                         // β, коэф
        d_out,                         // Out, выходные данные - нейроны для след слоя
        kernel);                       // Out, шаг до след Y (Y21 - Y11) 
   
    cublasGetMatrix(bsz, kernel, sizeof(snFloat), d_out, bsz, output, bsz);   
}
//
//bool fwdConvolution(size_t kernel, size_t fWidth, size_t fHeight, size_t stride,
//    snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* output){
//
//    size_t wStepByD = fWidth * fHeight,        // шаг весов по входу
//           wStepByK = wStepByD * insz.d,       // шаг весов по выходу
//           inStepByD = insz.w * insz.h,        // шаг вх слоя по входу
//           inStepByN = inStepByD * insz.d,     // шаг вх слоя по батчу
//           outStepByD = outsz.w * outsz.h,     // шаг вых слоя по выходу
//           outStepByN = outStepByD * outsz.d;  // шаг вых слоя по батчу
//
//    size_t shareStepByN = insz.d + kernel;     // для локализации памяти
//    snFloat* share = (snFloat*)calloc(shareStepByN * insz.n, sizeof(snFloat));
//    
//    memset(output, 0, outStepByN * insz.n * sizeof(snFloat));
//        
//    // по батчу
//#pragma omp parallel for
//    for (int n = 0; n < insz.n; ++n){
//
//        snFloat* inBuff = share + shareStepByN * n;
//        snFloat* outBuff = share + insz.d + shareStepByN * n;
//        
//        for (size_t p = 0; p < outStepByD; ++p){
//        
//            size_t ox = p % outsz.w, oy = p / outsz.w,
//                posW = ox * stride, posH = oy * stride;
//
//            memset(outBuff, 0, kernel * sizeof(snFloat));
//
//            // ядро свертки
//            for (size_t c = 0; c < (fWidth * fHeight); ++c){
//
//                size_t cx = c % fWidth, cy = c / fWidth;
//                snFloat* pIn = input + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;
//                snFloat* pW = weight + cx + cy * fWidth;
//
//                for (size_t d = 0; d < insz.d; ++d){
//                    inBuff[d] = *pIn;
//                    pIn += inStepByD;
//                }
//                            
//                // по всем вых слоям
//                for (size_t k = 0; k < kernel; ++k){
//                                        
//                    // по всем вх слоям
//                    snFloat cout = 0;
//                    for (size_t d = 0; d < insz.d; ++d){
//                        cout += inBuff[d] * (*pW);
//                        pW += wStepByD;
//                    }
//                    pW += 1;           // bias;
//                    outBuff[k] += cout;
//                }
//            }
//
//            snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
//            snFloat* pW = weight + wStepByK;
//
//            // по всем вых слоям
//            for (size_t k = 0; k < kernel; ++k){
//               
//                *pOut += outBuff[k] + *(pW + k); // + bias
//               
//                pW += wStepByK;
//                pOut += outStepByD;
//            }
//        }        
//    }
//
//   free(share);
//   return true;
//}
//
//bool fwdPooling(int type, size_t kernel, snSize insz, snFloat* input,
//    snSize outsz, snFloat* output, size_t* outputInx){
//
//    size_t inStepByD = insz.w * insz.h,           // шаг вх слоя по входу
//           inStepByN = inStepByD * insz.d,        // шаг вх слоя по батчу
//           outStepByD = outsz.w * outsz.h,        // шаг вых слоя по выходу
//           outStepByN = outStepByD * outsz.d,     // шаг вых слоя по батчу
//           kernelSz = kernel * kernel;
//   
//    size_t* shareI = (size_t*)calloc(insz.d * insz.n, sizeof(size_t));
//    snFloat* shareF = (snFloat*)calloc(insz.d * insz.n, sizeof(snFloat));
//
//    memset(output, 0, outStepByN * insz.n * sizeof(snFloat));
//    memset(outputInx, 0, outStepByN * insz.n * sizeof(snFloat));
//
//    if (type == 0){ // max
//
//        // по батчу
//#pragma omp parallel for
//        for (int n = 0; n < insz.n; ++n){
//
//            snFloat* outBuff = shareF + insz.d * n;
//            size_t* outInxBuff = shareI + insz.d * n;
//
//            for (size_t p = 0; p < outStepByD; ++p){
//
//                size_t ox = p % outsz.w, oy = p / outsz.w,
//                    posW = ox * kernel, posH = oy * kernel;
//
//                memset(outBuff, 0, insz.d * sizeof(snFloat));
//                memset(outInxBuff, 0, insz.d * sizeof(size_t));
//
//                // ядро свертки
//                for (size_t c = 0; c < kernelSz; ++c){
//
//                    size_t cx = c % kernel, cy = c / kernel;
//                    snFloat* pIn = input + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;
//
//                    // по всем вх слоям
//                    for (size_t d = 0; d < insz.d; ++d){
//                        snFloat val = *pIn;
//                        pIn += inStepByD;
//                        if (val > outBuff[d]){
//                            outBuff[d] = val;
//                            outInxBuff[d] = c;
//                        }
//                    }
//                }
//
//                snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
//                size_t* pOutInx = outputInx + ox + oy * outsz.w + n * outStepByN;
//
//                // по всем вых слоям
//                for (size_t k = 0; k < outsz.d; ++k){
//
//                    *pOut = outBuff[k];
//                    *pOutInx = outInxBuff[k];
//
//                    pOut += outStepByD;
//                    pOutInx += outStepByD;
//                }
//            }
//        }
//    }
//    else{ // mean
//
//        // по батчу
//#pragma omp parallel for
//        for (int n = 0; n < insz.n; ++n){
//
//            snFloat* outBuff = shareF + insz.d * n;
//          
//            for (size_t p = 0; p < outStepByD; ++p){
//
//                size_t ox = p % outsz.w, oy = p / outsz.w,
//                    posW = ox * kernel, posH = oy * kernel;
//
//                memset(outBuff, 0, insz.d * sizeof(snFloat));
//              
//                // ядро свертки
//                for (size_t c = 0; c < kernelSz; ++c){
//
//                    size_t cx = c % kernel, cy = c / kernel;
//                    snFloat* pIn = input + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;
//
//                    // по всем вх слоям
//                    for (size_t d = 0; d < insz.d; ++d){
//                        outBuff[d] += *pIn;
//                        pIn += inStepByD;
//                    }
//                }
//
//                snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
//
//                // по всем вых слоям
//                for (size_t k = 0; k < outsz.d; ++k){
//                    *pOut = outBuff[k] / kernelSz;
//                    pOut += outStepByD;
//                }
//            }
//        }
//    }
//   
//    free(shareI); 
//    free(shareF);
//    return true;
//}
//
//bool fwdBatchNorm(snSize insz, snFloat* in, snFloat* out, batchNorm prm){
//     
//    size_t inSz = insz.w * insz.h * insz.d, bsz = insz.n;
//
//    cublasHandle_t cuHandle = nullptr;
//    int sts = cublasCreate(&cuHandle);
//    if (sts != CUBLAS_STATUS_SUCCESS) {
//        fprintf(stderr, ("fwdBatchNorm CUBLAS initialization error: sts " + to_string(sts) + "\n").c_str());
//        return false;
//    }
//
//    snFloat* d_in = 0;
//    cudaMalloc(reinterpret_cast<void **>(&d_in), bsz * inSz * sizeof(snFloat));
//    cublasSetMatrix(bsz, inSz, sizeof(snFloat), in, inSz, d_in, inSz);
//
//    snFloat* d_onc = 0;
//    cudaMalloc(reinterpret_cast<void **>(&d_in), bsz * sizeof(snFloat));
//    cublasSetVector(bsz, sizeof(snFloat), prm.onc, 1, d_onc, 1);
//
//    snFloat *d_mean = 0;
//    cudaMalloc(reinterpret_cast<void **>(&d_mean), inSz * sizeof(snFloat));
//   
//    /// μ = 1/n * ∑x
//    float alpha = 1.0f / bsz;
//    float beta = 0.0f;
//    cublasSgemv(cuHandle,
//        CUBLAS_OP_T,
//        bsz,                          // x, строк - размер батча
//        inSz,                         // x, столбцов 
//        &alpha,                       // коэф
//        d_in,                         // x, данные
//        inSz,                         // x, шаг до след 
//        d_onc,                        // 1й вектор
//        1,                            // 1й вектор, шаг движения по вектору
//        &beta,                        // коэф
//        d_mean,                       // μ, результ
//        1);                           // μ, шаг до след
//   
//    cublasGetVector(inSz, sizeof(snFloat), d_mean, 1, prm.onc, 1);
//
//    cudaFree(d_in);
//    cudaFree(d_onc);
//    cudaFree(d_mean);
//    
//    cublasDestroy(cuHandle);
//
//    /// varce = sqrt(∑xx - mean^2 + e)
//    for (size_t i = 0; i < inSz; ++i){
//
//        snFloat* cin = in + i, srq = 0.F;
//        for (size_t j = 0; j < bsz; ++j){
//            srq += cin[0] * cin[0];
//            cin += inSz;
//        }
//        prm.varce[i] = sqrt(srq / bsz - prm.mean[i] * prm.mean[i] + 0.00001F);
//    }
//      
//    /// norm = (in - mean) / varce
//    /// y = ^x * γ + β
//    for (size_t j = 0; j < bsz; ++j){
//
//        snFloat* cin = in + j * inSz, *cout = out + j * inSz, *norm = prm.norm + j * inSz;
//
//        for (size_t i = 0; i < inSz; ++i){                        
//            norm[i] = (cin[i] - prm.mean[i]) / prm.varce[i];
//            cout[i] = norm[i] * prm.scale[i] + prm.schift[i];
//        }
//    }  
//
//    return true;
//}

/// батч нормализация прямой проход CUDA    
void batchNormForwardCUDA(SN_Base::snSize insz,
    SN_Base::snFloat* in,
    SN_Base::snFloat* out,
    SN_Base::batchNorm,
    std::map<std::string, void*>&){


}

#endif // SN_CUDA

//
//#define m 3 // a - mxk matrix
//#define n 4 // b - kxn matrix
//#define k 5 // c - mxn matrix
//
//
//
//cudaError_t cudaStat; // cudaMalloc status
//cublasStatus_t stat; // CUBLAS functions status
//cublasHandle_t handle; // CUBLAS context   
//
//float* a = (float*)malloc(m*k* sizeof(float)); // host memory for a
//float* b = (float*)malloc(k*n* sizeof(float)); // host memory for b
//float* c = (float*)malloc(m*n* sizeof(float)); // host memory for c
//
//// define an mxk matrix a column by column
//printf("a:\n");
//int ind = 0;                             // a:
//for (int i = 0; i < m; ++i){                   // 0  1  2  3  4
//    for (int j = 0; j < k; ++j){               // 5  6  7  8  9
//        a[j + i * k] = (float)++ind;           // 10 11 12 13 14
//        printf(" %5.0f", a[j + i * k]);
//    }
//    printf("\n");
//}
//
//// define a kxn matrix b column by column   
//printf("b:\n");
//ind = 1;                                 // b: 
//for (int i = 0; i < k; ++i){                   // 1  2  3  4
//    for (int j = 0; j < n; ++j){               // 5  6  7  8
//        b[j + i * n] = (float)++ind;           // 9  10 11 12
//        printf(" %5.0f", b[j + i * n]);       // 13 14 15 16
//    }                                          // 17 18 19 20 
//    printf("\n");
//}
//
//// on the device
//float * d_a; // d_a - a on the device
//float * d_b; // d_b - b on the device
//float * d_c; // d_c - c on the device
//cudaStat = cudaMalloc((void **)& d_a, m*k* sizeof(*a)); // device
//
//// memory alloc for a
//cudaStat = cudaMalloc((void **)& d_b, k*n* sizeof(*b)); // device
//
//// memory alloc for b
//cudaStat = cudaMalloc((void **)& d_c, m*n* sizeof(*c)); // device
//
//// memory alloc for c
//stat = cublasCreate(&handle); // initialize CUBLAS context
//
//// copy matrices from the host to the device
//stat = cublasSetMatrix(m, k, sizeof(*a), a, m, d_a, m); //a -> d_a
//stat = cublasSetMatrix(k, n, sizeof(*b), b, k, d_b, k); //b -> d_b
//stat = cublasSetMatrix(m, n, sizeof(*c), c, m, d_c, m); //c -> d_c
//
//// a - mxk matrix
//// b - kxn matrix
//// c - mxn matrix
//
//float al = 1.0f;   // al =1
//float bet = 0.0f;  // bet =1
//stat = cublasSgemm(handle,
//    CUBLAS_OP_N,
//    CUBLAS_OP_N,
//    n,   // строки 1й
//    m,   // столбцы 2й
//    k,   // столбцы 1й
//    &al,
//    d_b,
//    n,
//    d_a,
//    k,
//    &bet,
//    d_c,
//    n);
//
//stat = cublasGetMatrix(m, n, sizeof(*c), d_c, m, c, m); // cp d_c - >c
//printf("c after Sgemm :\n");
//for (int i = 0; i < m; i++){
//    for (int j = 0; j < n; j++){
//        printf(" %7.0f", c[j + i * n]);
//    }
//    printf("\n");
//}
//
//bool ff = false;
//#undef m
//#undef n
//#undef k