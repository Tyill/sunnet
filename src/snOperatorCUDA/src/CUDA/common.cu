//
// sunnet project
// Copyright (C) 2018 by Contributors <https://github.com/Tyill/sunnet>
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

#include <cuda_runtime.h>
#include "../stdafx.h"

using namespace SN_Base;

void cuSetDeviceId(int id){

   cuAssert(cudaSetDevice(id));
}


__global__ void cuMemSetInf(snSize insz, snFloat* in, snFloat val){

    size_t inStepByD = insz.w * insz.h,     // step out by input
        inStepByN = inStepByD * insz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    snFloat* cin = in + inStepByN * blockIdx.y + inStepByD * blockIdx.x;

    unsigned int i = threadIdx.x;
    while (i < inStepByD){

        cin[i] = val;

        i += blockDim.x;
    }
}

void cuMemSet(const snSize& sz, snFloat* data, snFloat val){

    if (val == 0.F){
        cuAssert(cudaMemset(data, 0, sz.size() * sizeof(snFloat)));
    }
    else{

        dim3 dimBlock(128);
        dim3 dimGrid(int(sz.d), int(sz.n));

        cuMemSetInf << <dimGrid, dimBlock >> >(sz, data, val);
    }
}

snFloat* cuMemAlloc(const snSize& sz, snFloat initVal){

    snFloat* mem = nullptr;
    cuAssert(cudaMalloc(&mem, sz.size() * sizeof(snFloat)));

    cuMemSet(sz, mem, initVal);

    return mem;
}

snFloat* cuMemRealloc(const snSize& csz, const snSize& nsz, snFloat* data, snFloat initVal){

    size_t tcsz = csz.size(),
           tnsz = nsz.size();

    ASSERT_MESS(tnsz > 0, "");

    if (tcsz < tnsz){

        snFloat* mem = nullptr;
        cuAssert(cudaMalloc(&mem, tnsz * sizeof(snFloat)));

        cuMemSet(nsz, mem, initVal);
        
        if (data){
            if (tcsz > 0)
               cuAssert(cudaMemcpy(mem, data, tcsz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
            cuAssert(cudaFree(data));
        }
        data = mem;
    }
   
    return data;
}

void cuMemCpyCPU2GPU(const snSize& sz, SN_Base::snFloat* dstGPU, SN_Base::snFloat* srcCPU){
  
    cuAssert(cudaMemcpy(dstGPU, srcCPU, sz.size() * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyHostToDevice));
}
                          
void cuMemCpyGPU2CPU(const snSize& sz, SN_Base::snFloat* dstCPU, SN_Base::snFloat* srcGPU){

    cuAssert(cudaMemcpy(dstCPU, srcGPU, sz.size() * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}
                          
void cuMemCpyGPU2GPU(const snSize& sz, SN_Base::snFloat* dstGPU, SN_Base::snFloat* srcGPU, bool isAsync){

    if (isAsync){
        cuAssert(cudaMemcpyAsync(dstGPU, srcGPU, sz.size() * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice, 0));
    }
    else{
        cuAssert(cudaMemcpy(dstGPU, srcGPU, sz.size() * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    }
}

void cuMemFree(snFloat* data){

    cuAssert(cudaFree(data));
}