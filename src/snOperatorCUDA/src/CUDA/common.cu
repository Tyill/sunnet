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
#pragma once

#include <cuda_runtime.h>
#include "../stdafx.h"

using namespace SN_Base;

void cuSetDeviceId(int id){

   cuAssert(cudaSetDevice(id));
}

snFloat* cuMemAlloc(size_t sz, int initVal){

    snFloat* mem = nullptr;
    cuAssert(cudaMalloc(&mem, sz * sizeof(snFloat)));

    cuAssert(cudaMemset(mem, initVal, sz * sizeof(snFloat)));

    return mem;
}

snFloat* cuMemRealloc(size_t csz, size_t nsz, snFloat* data, int initVal){

    ASSERT_MESS(nsz > 0, "");

    if (csz < nsz){

        snFloat* mem = nullptr;
        cuAssert(cudaMalloc(&mem, nsz * sizeof(snFloat)));

        cuAssert(cudaMemset(mem, initVal, nsz * sizeof(snFloat)));
        
        if (data){
            if (csz > 0)
               cuAssert(cudaMemcpy(mem, data, csz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
            cuAssert(cudaFree(data));
        }
        data = mem;
    }
   
    return data;
}

void cuMemSet(size_t sz, snFloat* data, int val){
      
    cuAssert(cudaMemset(data, val, sz * sizeof(snFloat)));
}

void cuMemCpyCPU2GPU(size_t sz, SN_Base::snFloat* dstGPU, SN_Base::snFloat* srcCPU){
  
    cuAssert(cudaMemcpy(dstGPU, srcCPU, sz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyHostToDevice));
}
                          
void cuMemCpyGPU2CPU(size_t sz, SN_Base::snFloat* dstCPU, SN_Base::snFloat* srcGPU){

    cuAssert(cudaMemcpy(dstCPU, srcGPU, sz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}
                          
void cuMemCpyGPU2GPU(size_t sz, SN_Base::snFloat* dstGPU, SN_Base::snFloat* srcGPU, bool isAsync){
 
    if (isAsync)
       cuAssert(cudaMemcpyAsync(dstGPU, srcGPU, sz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice, 0));
    else
       cuAssert(cudaMemcpy(dstGPU, srcGPU, sz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}

void cuMemFree(snFloat* data){

    cuAssert(cudaFree(data));
}