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

void FullyConnected::iniParamCUDA(const snSize& insz, size_t kernel, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    size_t ida = insz.w * insz.h * insz.d + 1, bsz = insz.n;

    if (gpuPrm.find("hcuBLAS") == gpuPrm.end()){
        
        cublasHandle_t cuHandle = nullptr;
        cuCHECK(cublasCreate(&cuHandle));

        gpuPrm["hcuBLAS"] = cuHandle;
          
        gpuPrm["d_in"] = 0;
        gpuPrm["d_w"] = 0;
        gpuPrm["d_out"] = 0;
        gpuPrm["d_grout"] = 0;
        gpuPrm["d_dw"] = 0;

        if (!gpuClearMem_){
            cuCHECK(cudaMalloc(&gpuPrm["d_in"], bsz * ida * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm["d_w"], ida * kernel * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm["d_out"], bsz * kernel * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm["d_grout"], bsz * (ida - 1) * sizeof(snFloat)));
            cuCHECK(cudaMalloc(&gpuPrm["d_dw"], ida * kernel * sizeof(snFloat)));
        }
    }
    else if (!gpuClearMem_){
          
        cuCHECK(cudaFree(gpuPrm["d_in"]));    gpuPrm["d_in"] = 0;
        cuCHECK(cudaFree(gpuPrm["d_w"]));     gpuPrm["d_w"] = 0;
        cuCHECK(cudaFree(gpuPrm["d_out"]));   gpuPrm["d_out"] = 0;
        cuCHECK(cudaFree(gpuPrm["d_grout"])); gpuPrm["d_grout"] = 0;
        cuCHECK(cudaFree(gpuPrm["d_dw"]));    gpuPrm["d_dw"] = 0;

        cuCHECK(cudaMalloc(&gpuPrm["d_in"], bsz * ida * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_w"], ida * kernel * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_out"], bsz * kernel * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_grout"], bsz * (ida - 1) * sizeof(snFloat)));
        cuCHECK(cudaMalloc(&gpuPrm["d_dw"], ida * kernel * sizeof(snFloat)));
    }
}
         
void FullyConnected::freeParamCUDA(map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    if (gpuPrm.find("hcuBLAS") == gpuPrm.end()) return;

    cublasDestroy((cublasHandle_t)gpuPrm["hcuBLAS"]);

    if (!gpuClearMem_){
        for (auto p : gpuPrm)
            if (p.first != "hcuBLAS")  cudaFree(p.second);
    }
}

void FullyConnected::forwardCUDA(size_t kernel, const snSize& insz, snFloat* input, snFloat* weight, snFloat* output, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    if (gpuPrm.find("hcuBLAS") == gpuPrm.end()) return;

    cublasHandle_t hcuBLAS = (cublasHandle_t)gpuPrm["hcuBLAS"];

    int ida = int(insz.w * insz.h * insz.d + 1), bsz = int(insz.n), krn = int(kernel);
   
    snFloat *d_in  = (snFloat*)gpuPrm["d_in"],
            *d_w   = (snFloat*)gpuPrm["d_w"], 
            *d_out = (snFloat*)gpuPrm["d_out"];
   
    if (gpuClearMem_){
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_in), bsz * ida * sizeof(snFloat)));
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_w), ida * kernel * sizeof(snFloat)));
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bsz * kernel * sizeof(snFloat)));
    }

    cuCHECK(cublasSetMatrix(bsz, ida, sizeof(snFloat), input, bsz, d_in, bsz));
    
    cuCHECK(cublasSetMatrix(ida, krn, sizeof(snFloat), weight, ida, d_w, ida));
   
    // Out = α * W * In + βC
    // In - матрица вход данных - значения с предыд слоя
    // W - матрица весов
    // Out - матрица выход данных
    float alpha = 1.0f, beta = 0.0f;
    cuCHECK(cublasSgemm(hcuBLAS,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        krn,                           // W, столбцов, кол-во скрытых нейронов 
        bsz,                           // In, строк, кол-во изобр в батче
        ida,                           // In, столбцов, В М - строк, кол-во вх нейронов - размер одного изображения из батча. (+1 - X0)                   
        &alpha,                        // α, коэф
        d_w,                           // W, веса
        krn,                           // W, шаг до след W (W21 - W11)
        d_in,                          // In, вх данные - нейроны пришедшие с предыд слоя
        ida,                           // In, шаг до след X (X21 - X11)  
        &beta,                         // β, коэф
        d_out,                         // Out, выходные данные - нейроны для след слоя
        krn));                         // Out, шаг до след Y (Y21 - Y11) 
    
    cuCHECK(cublasGetMatrix(bsz, krn, sizeof(snFloat), d_out, bsz, output, bsz));

    if (gpuClearMem_){
        cuCHECK(cudaFree(d_in));
        cuCHECK(cudaFree(d_w));
        cuCHECK(cudaFree(d_out));
    }
}

void FullyConnected::backwardCUDA_GW(size_t kernel, snFloat* weight,
    const snSize& insz, snFloat* input, snFloat* gradIn, snFloat* gradOut, snFloat* dWOut, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    if (gpuPrm.find("hcuBLAS") == gpuPrm.end()) return;

    cublasHandle_t hcuBLAS = (cublasHandle_t)gpuPrm["hcuBLAS"];

    int ida = int(insz.w * insz.h * insz.d + 1), bsz = int(insz.n), krn = int(kernel);

    snFloat* d_grin = (snFloat*)gpuPrm["d_out"],
           * d_in = (snFloat*)gpuPrm["d_in"],
           * d_w = (snFloat*)gpuPrm["d_w"],
           * d_dw = (snFloat*)gpuPrm["d_dw"],
           * d_grout = (snFloat*)gpuPrm["d_grout"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_in), bsz * ida * sizeof(snFloat)));          
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_w), ida * kernel * sizeof(snFloat)));        
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_grin), bsz * kernel * sizeof(snFloat)));    
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_grout), bsz * (ida - 1) * sizeof(snFloat)));
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_dw), ida * kernel * sizeof(snFloat)));       
    }

    cuCHECK(cublasSetMatrix(bsz, ida, sizeof(snFloat), input, bsz, d_in, bsz));
      
    cuCHECK(cublasSetMatrix(bsz, krn, sizeof(snFloat), gradIn, bsz, d_grin, bsz));

    // Градиент по весам
    // dW = αIn^T * GrIn + βdW
    // In - матрица вход данных с предыд слоя
    // GrIn - матрица градиентов со след слоя
    float alpha = 1.0F / insz.n, beta = 0.0f;
    cuCHECK(cublasSgemm(hcuBLAS,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        krn,                     // GrIn, столбцов, кол-во скрытых нейронов 
        ida,                     // In, столбцов, кол-во вх значений (+1 - X0)      
        bsz,                     // In, строк, кол-во изобр в батче
        &alpha,                  // α, коэф                 
        d_grin,                  // GrIn - градиент пришедший со след слоя
        krn,                     // GrIn - шаг до след
        d_in,                    // In, вх данные - нейроны пришедшие с предыд слоя
        ida,                     // In, шаг до след X (X21 - X11)  
        &beta,                   // β, коэф                 
        d_dw,                    // dW, выходные данные - градиент по весам                 
        krn));                   // dW, шаг до след

    cuCHECK(cublasGetMatrix(ida, krn, sizeof(snFloat), d_dw, ida, dWOut, ida));
         
    cuCHECK(cublasSetMatrix(ida - 1, krn, sizeof(snFloat), weight + kernel, ida - 1, d_w, ida - 1));

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
        krn,                     // GrIn, столбцов, кол-во скрытых нейронов 
        &alpha,                  // α, коэф                                  
        d_w,                     // W - веса
        krn,                     // W - шаг до след
        d_grin,                  // GrIn, вх данные - нейроны пришедшие с предыд слоя
        krn,                     // GrIn, шаг до след 
        &beta,                   // β, коэф                 
        d_grout,                 // GrOut, выходной градиент для пред слоя                                   
        ida - 1));               // GrOut, шаг до след
      
    cuCHECK(cublasGetMatrix(bsz, ida - 1, sizeof(snFloat), d_grout, bsz, gradOut, bsz));
 
    if (gpuClearMem_){
        cuCHECK(cudaFree(d_in));
        cuCHECK(cudaFree(d_w));
        cuCHECK(cudaFree(d_grin));
        cuCHECK(cudaFree(d_grout));
        cuCHECK(cudaFree(d_dw));
    }
}

void FullyConnected::backwardCUDA_G(size_t kernel, snFloat* weight, const snSize& insz, snFloat* gradIn, snFloat* gradOut, map<string, void*>& gpuPrm){
    cudaSetDevice(gpuDeviceId_);

    if (gpuPrm.find("hcuBLAS") == gpuPrm.end()) return;

    cublasHandle_t hcuBLAS = (cublasHandle_t)gpuPrm["hcuBLAS"];

    int ida = int(insz.w * insz.h * insz.d + 1), bsz = int(insz.n), krn = int(kernel);

    snFloat* d_grin = (snFloat*)gpuPrm["d_out"],
           * d_w = (snFloat*)gpuPrm["d_w"],
           * d_grout = (snFloat*)gpuPrm["d_grout"];

    if (gpuClearMem_){
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_w), ida * kernel * sizeof(snFloat)));
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_grin), bsz * kernel * sizeof(snFloat)));
        cuCHECK(cudaMalloc(reinterpret_cast<void**>(&d_grout), bsz * (ida - 1) * sizeof(snFloat)));
    }

    cuCHECK(cublasSetMatrix(bsz, krn, sizeof(snFloat), gradIn, bsz, d_grin, bsz));

    cuCHECK(cublasSetMatrix(ida - 1, krn, sizeof(snFloat), weight + kernel, ida - 1, d_w, ida - 1));

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
        krn,                     // GrIn, столбцов, кол-во скрытых нейронов 
        &alpha,                  // α, коэф                                  
        d_w,                     // W - веса
        krn,                     // W - шаг до след
        d_grin,                  // GrIn, вх данные - нейроны пришедшие с предыд слоя
        krn,                     // GrIn, шаг до след 
        &beta,                   // β, коэф                 
        d_grout,                 // GrOut, выходной градиент для пред слоя                                   
        ida - 1));               // GrOut, шаг до след

    cuCHECK(cublasGetMatrix(bsz, ida - 1, sizeof(snFloat), d_grout, bsz, gradOut, bsz));

    if (gpuClearMem_){
        cuCHECK(cudaFree(d_w));
        cuCHECK(cudaFree(d_grin));
        cuCHECK(cudaFree(d_grout));
    }
}

#endif 