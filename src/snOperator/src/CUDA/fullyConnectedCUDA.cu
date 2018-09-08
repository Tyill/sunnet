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
#include "SNOperator/src/Operator/fullyConnected.h"

using namespace std;
using namespace SN_Base;
         
#ifndef cuCHECK
#define cuCHECK(func) if (func != 0){ ERROR_MESS("CUDA error: " + cudaGetErrorString(cudaGetLastError())); return;}
#endif

void FullyConnected::iniParamCUDA(snSize insz, size_t kernel, map<string, void*>& gpuPrm){
    
    size_t ida = insz.w * insz.h * insz.d + 1, bsz = insz.n;

    if (gpuPrm.find("hcuBLAS") == gpuPrm.end()){
        
        cublasHandle_t cuHandle = nullptr;
        cuCHECK(cublasCreate(&cuHandle));

        gpuPrm["hcuBLAS"] = cuHandle;
                            
        snFloat* d_in_fwd = 0, *d_w_fwd = 0, *d_out_fwd = 0;
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_in_fwd), bsz * ida * sizeof(snFloat)));     gpuPrm["d_in_fwd"] = d_in_fwd;
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_w_fwd), ida * kernel * sizeof(snFloat)));   gpuPrm["d_w_fwd"] = d_w_fwd;
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_out_fwd), bsz * kernel * sizeof(snFloat))); gpuPrm["d_out_fwd"] = d_out_fwd;
    }
    else{
        snFloat* d_in_fwd  = (snFloat*)gpuPrm["d_in_fwd"],
               * d_w_fwd   = (snFloat*)gpuPrm["d_w_fwd"],
               * d_out_fwd = (snFloat*)gpuPrm["d_out_fwd"];            

        cuCHECK(cudaFree(d_in_fwd));  cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_in_fwd), bsz * ida * sizeof(snFloat)));     gpuPrm["d_in_fwd"] = d_in_fwd;
        cuCHECK(cudaFree(d_w_fwd));   cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_w_fwd), ida * kernel * sizeof(snFloat)));   gpuPrm["d_w_fwd"] = d_w_fwd;
        cuCHECK(cudaFree(d_out_fwd)); cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_out_fwd), bsz * kernel * sizeof(snFloat))); gpuPrm["d_out_fwd"] = d_out_fwd;
    }
}
         
void FullyConnected::freeParamCUDA(map<string, void*>& gpuPrm){
    
    if (gpuPrm.find("hcuBLAS") == gpuPrm.end()) return;

    cublasDestroy((cublasHandle_t)gpuPrm["hcuBLAS"]);

    for (auto p : gpuPrm)
        if (p.first != "hcuBLAS")  cudaFree(p.second);
}

void FullyConnected::forwardCUDA(size_t kernel, snSize insz, snFloat* input, snFloat* weight, snFloat* output, map<string, void*>& gpuPrm){
     
    if (gpuPrm.find("hcuBLAS") == gpuPrm.end()) return;

    size_t ida = insz.w * insz.h * insz.d + 1, bsz = insz.n;
   
    snFloat *d_in  = (snFloat*)gpuPrm["d_in_FWD"],
            *d_w   = (snFloat*)gpuPrm["d_w_FWD"], 
            *d_out = (snFloat*)gpuPrm["d_out_FWD"];
   
    cuCHECK(cublasSetMatrix(bsz, ida, sizeof(snFloat), input, bsz, d_in, bsz));
    
    cuCHECK(cublasSetMatrix(ida, kernel, sizeof(snFloat), weight, ida, d_w, ida));
   
    // Out = α * W * In + βC
    // In - матрица вход данных - значения с предыд слоя
    // W - матрица весов
    // Out - матрица выход данных
    float alpha = 1.0f, beta = 0.0f;
    cuCHECK(cublasSgemm((cublasHandle_t)gpuPrm["hcuBLAS"],
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
        kernel));                      // Out, шаг до след Y (Y21 - Y11) 
    
    cuCHECK(cublasGetMatrix(bsz, kernel, sizeof(snFloat), d_out, bsz, output, bsz));
   
}

void FullyConnected::backwardCUDA_GW(size_t kernel, snFloat* weight,
    snSize insz, snFloat* input, snFloat* gradIn, snFloat* gradOut, snFloat* dWOut, map<string, void*>&){

    backwardCPU_GW(kernel, weight,
        insz, input, gradIn, gradOut, dWOut);
}

void FullyConnected::backwardCUDA_G(size_t kernel, snFloat* weight, snSize insz, snFloat* gradIn, snFloat* gradOut, map<string, void*>&){


}


#endif 


// SN_CUDA

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