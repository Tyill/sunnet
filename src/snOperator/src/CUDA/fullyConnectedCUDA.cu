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

#include "Lib/OpenBLAS/cblas.h"
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
                            
        snFloat* d_in = 0, *d_w = 0, *d_out = 0, *d_grout = 0, *d_dw = 0;
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_in), bsz * ida * sizeof(snFloat)));          gpuPrm["d_in"] = d_in;
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_w), ida * kernel * sizeof(snFloat)));        gpuPrm["d_w"] = d_w;
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bsz * kernel * sizeof(snFloat)));      gpuPrm["d_out"] = d_out;
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_grout), bsz * (ida - 1) * sizeof(snFloat))); gpuPrm["d_grout"] = d_grout;
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_dw), ida * kernel * sizeof(snFloat)));       gpuPrm["d_dw"] = d_dw;
    }
    else{
        snFloat* d_in    = (snFloat*)gpuPrm["d_in"],
               * d_w     = (snFloat*)gpuPrm["d_w"],
               * d_dw    = (snFloat*)gpuPrm["d_dw"],
               * d_out   = (snFloat*)gpuPrm["d_out"],
               * d_grout = (snFloat*)gpuPrm["d_grout"];            

        cuCHECK(cudaFree(d_in));    cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_in), bsz * ida * sizeof(snFloat)));          gpuPrm["d_in"] = d_in;
        cuCHECK(cudaFree(d_w));     cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_w), ida * kernel * sizeof(snFloat)));        gpuPrm["d_w"] = d_w;
        cuCHECK(cudaFree(d_out));   cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bsz * kernel * sizeof(snFloat)));      gpuPrm["d_out"] = d_out;
        cuCHECK(cudaFree(d_grout)); cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_grout), bsz * (ida - 1) * sizeof(snFloat))); gpuPrm["d_grout"] = d_grout;
        cuCHECK(cudaFree(d_dw));    cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_dw), ida * kernel * sizeof(snFloat)));       gpuPrm["d_dw"] = d_dw;
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

    cublasHandle_t hcuBLAS = (cublasHandle_t)gpuPrm["hcuBLAS"];

    size_t ida = insz.w * insz.h * insz.d + 1, bsz = insz.n;
   
    snFloat *d_in  = (snFloat*)gpuPrm["d_in"],
            *d_w   = (snFloat*)gpuPrm["d_w"], 
            *d_out = (snFloat*)gpuPrm["d_out"];
   
    cuCHECK(cublasSetMatrix(bsz, ida, sizeof(snFloat), input, bsz, d_in, bsz));
    
    cuCHECK(cublasSetMatrix(ida, kernel, sizeof(snFloat), weight, ida, d_w, ida));
   
    // Out = α * W * In + βC
    // In - матрица вход данных - значения с предыд слоя
    // W - матрица весов
    // Out - матрица выход данных
    float alpha = 1.0f, beta = 0.0f;
    cuCHECK(cublasSgemm(hcuBLAS,
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
    snSize insz, snFloat* input, snFloat* gradIn, snFloat* gradOut, snFloat* dWOut, map<string, void*>& gpuPrm){
       
    if (gpuPrm.find("hcuBLAS") == gpuPrm.end()) return;

    cublasHandle_t hcuBLAS = (cublasHandle_t)gpuPrm["hcuBLAS"];

    size_t ida = insz.w * insz.h * insz.d + 1, bsz = insz.n;

    snFloat* d_grin = (snFloat*)gpuPrm["d_out"],
           * d_in = (snFloat*)gpuPrm["d_in"],
           * d_w = (snFloat*)gpuPrm["d_w"],
           * d_dw = (snFloat*)gpuPrm["d_dw"],
           * d_grout = (snFloat*)gpuPrm["d_grout"];

    cuCHECK(cublasSetMatrix(bsz, ida, sizeof(snFloat), input, bsz, d_in, bsz));
      
    cuCHECK(cublasSetMatrix(bsz, kernel, sizeof(snFloat), gradIn, bsz, d_grin, bsz));

    // Градиент по весам
    // dW = αIn^T * GrIn + βdW
    // In - матрица вход данных с предыд слоя
    // GrIn - матрица градиентов со след слоя
    float alpha = 1.0F / insz.n, beta = 0.0f;
    cuCHECK(cublasSgemm(hcuBLAS,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        kernel,                  // GrIn, столбцов, кол-во скрытых нейронов 
        ida,                     // In, столбцов, кол-во вх значений (+1 - X0)      
        bsz,                     // In, строк, кол-во изобр в батче
        &alpha,                  // α, коэф                 
        d_grin,                  // GrIn - градиент пришедший со след слоя
        kernel,                  // GrIn - шаг до след
        d_in,                    // In, вх данные - нейроны пришедшие с предыд слоя
        ida,                     // In, шаг до след X (X21 - X11)  
        &beta,                   // β, коэф                 
        d_dw,                    // dW, выходные данные - градиент по весам                 
        kernel));                // dW, шаг до след

    cuCHECK(cublasGetMatrix(ida, kernel, sizeof(snFloat), d_dw, ida, dWOut, ida));

     
    cuCHECK(cublasSetMatrix(ida - 1, kernel, sizeof(snFloat), weight + kernel, ida - 1, d_w, ida - 1));

    //// Градиент для предыд слоя
    //// GrOut = αGrIn * W^T + βGrOut
    //// GrIn - матрица градиентов со след слоя
    //// W - веса
    alpha = 1.F;
    cuCHECK(cublasSgemm(hcuBLAS,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        ida - 1,                 // W, столбцов, кол-во вх значений (+1 - X0)     
        bsz,                     // W, строк, кол-во изобр в батче    
        kernel,                  // GrIn, столбцов, кол-во скрытых нейронов 
        &alpha,                  // α, коэф                                  
        d_w,                     // W - веса
        kernel,                  // W - шаг до след
        d_grin,                  // GrIn, вх данные - нейроны пришедшие с предыд слоя
        kernel,                  // GrIn, шаг до след 
        &beta,                   // β, коэф                 
        d_grout,                 // GrOut, выходной градиент для пред слоя                                   
        ida - 1));               // GrOut, шаг до след
      
    cuCHECK(cublasGetMatrix(bsz, ida - 1, sizeof(snFloat), d_grout, bsz, gradOut, bsz));
    
}

void FullyConnected::backwardCUDA_G(size_t kernel, snFloat* weight, snSize insz, snFloat* gradIn, snFloat* gradOut, map<string, void*>& gpuPrm){

    if (gpuPrm.find("hcuBLAS") == gpuPrm.end()) return;

    cublasHandle_t hcuBLAS = (cublasHandle_t)gpuPrm["hcuBLAS"];

    size_t ida = insz.w * insz.h * insz.d + 1, bsz = insz.n;

    snFloat* d_grin = (snFloat*)gpuPrm["d_out"],
           * d_w = (snFloat*)gpuPrm["d_w"],
           * d_grout = (snFloat*)gpuPrm["d_grout"];

    cuCHECK(cublasSetMatrix(bsz, kernel, sizeof(snFloat), gradIn, bsz, d_grin, bsz));

    cuCHECK(cublasSetMatrix(ida - 1, kernel, sizeof(snFloat), weight + kernel, ida - 1, d_w, ida - 1));

    //// Градиент для предыд слоя
    //// GrOut = αGrIn * W^T + βGrOut
    //// GrIn - матрица градиентов со след слоя
    //// W - веса
    float alpha = 1.0F, beta = 0.0f;
    cuCHECK(cublasSgemm(hcuBLAS,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        ida - 1,                 // W, столбцов, кол-во вх значений (+1 - X0)     
        bsz,                     // W, строк, кол-во изобр в батче    
        kernel,                  // GrIn, столбцов, кол-во скрытых нейронов 
        &alpha,                  // α, коэф                                  
        d_w,                     // W - веса
        kernel,                  // W - шаг до след
        d_grin,                  // GrIn, вх данные - нейроны пришедшие с предыд слоя
        kernel,                  // GrIn, шаг до след 
        &beta,                   // β, коэф                 
        d_grout,                 // GrOut, выходной градиент для пред слоя                                   
        ida - 1));               // GrOut, шаг до след

    cuCHECK(cublasGetMatrix(bsz, ida - 1, sizeof(snFloat), d_grout, bsz, gradOut, bsz));
}

#endif 