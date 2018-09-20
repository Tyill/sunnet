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
#include "snOperator/src/Operator/fullyConnected.h"
#include <omp.h>  

using namespace std;
using namespace SN_Base;


void FullyConnected::forwardCPU(size_t kernel, const snSize& insz, snFloat* input, snFloat* weight, snFloat* output){

    // Out = α * In * W + βC
    // In - матрица вход данных - значения с предыд слоя
    // W - матрица весов
    // Out - матрица выход данных
    cblas_sgemm(CBLAS_ORDER::CblasRowMajor,
        CBLAS_TRANSPOSE::CblasNoTrans,
        CBLAS_TRANSPOSE::CblasNoTrans,
        insz.n,                        // In, строк, кол-во изобр в батче
        kernel,                        // W, столбцов, кол-во скрытых нейронов 
        insz.w * insz.h * insz.d + 1,  // In, столбцов, В М - строк, кол-во вх нейронов - размер одного изображения из батча. (+1 - X0)                   
        1.0F,                          // α, коэф
        input,                         // In, вх данные - нейроны пришедшие с предыд слоя
        insz.w * insz.h * insz.d + 1,  // In, шаг до след X (X21 - X11) 
        weight,                        // W, веса
        kernel,                        // W, шаг до след W (W21 - W11) 
        0.0,                           // β, коэф
        output,                        // Out, выходные данные - нейроны для след слоя
        kernel);                       // Out, шаг до след Y (Y21 - Y11) 
}

void FullyConnected::backwardCPU_GW(size_t kernel, snFloat* weight,
    const snSize& insz, snFloat* input, snFloat* gradIn, snFloat* gradOut, snFloat* dWOut){

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

void FullyConnected::backwardCPU_G(size_t kernel, snFloat* weight, const snSize& insz, snFloat* gradIn, snFloat* gradOut){

    size_t imSz = insz.w * insz.h * insz.d + 1;

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


#ifndef SN_CUDA

void FullyConnected::iniParamCUDA(const snSize& insz, size_t kernel, std::map<std::string, void*>& auxPrm){
    ERROR_MESS("CUDA non compiler");
}

void FullyConnected::freeParamCUDA(std::map<std::string, void*>& auxPrm){
    ERROR_MESS("CUDA non compiler");
}

void FullyConnected::forwardCUDA(size_t kernel, const snSize& insz, snFloat* input, snFloat* weight, snFloat* output, std::map<std::string, void*>& auxPrm){
    ERROR_MESS("CUDA non compiler");
}

void FullyConnected::backwardCUDA_GW(size_t kernel, snFloat* weight,
    const snSize& insz, snFloat* input, snFloat* gradIn, snFloat* gradOut, snFloat* dWOut, std::map<std::string, void*>&){
    ERROR_MESS("CUDA non compiler");
}

void FullyConnected::backwardCUDA_G(size_t kernel, snFloat* weight, const snSize& insz, snFloat* gradIn, snFloat* gradOut, std::map<std::string, void*>&){
    ERROR_MESS("CUDA non compiler");
}


#endif


#ifndef SN_OpenCL

void FullyConnected::iniParamOCL(const SN_Base::snSize& insz, size_t kernel, std::map<std::string, void*>& auxPrm){
    ERROR_MESS("OpenCL non compiler");
}

void FullyConnected::freeParamOCL(std::map<std::string, void*>& auxPrm){
    ERROR_MESS("OpenCL non compiler");
}

void FullyConnected::forwardOCL(size_t kernel, const snSize& insz, snFloat* input, snFloat* weight, snFloat* output, std::map<std::string, void*>& auxPrm){
    ERROR_MESS("OpenCL non compiler");
}

void FullyConnected::backwardOCL_GW(size_t kernel, snFloat* weight,
    const snSize& insz, snFloat* input, snFloat* gradIn, snFloat* gradOut, snFloat* dWOut, std::map<std::string, void*>&){
    ERROR_MESS("OpenCL non compiler");
}

void FullyConnected::backwardOCL_G(size_t kernel, snFloat* weight, const snSize& insz, snFloat* gradIn, snFloat* gradOut, std::map<std::string, void*>&){
    ERROR_MESS("OpenCL non compiler");
}


#endif