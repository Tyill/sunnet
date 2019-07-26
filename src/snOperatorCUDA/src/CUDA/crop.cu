
#include <cuda_runtime.h>

#include "../stdafx.h"
#include "../structurs.h"

using namespace SN_Base;


__global__ void cropIn2Out(roi roi, snSize srcSz, snFloat* in, snFloat* out){

    size_t srcStp = srcSz.w * srcSz.h,
           dstStp = roi.w * roi.h;
      
    in += roi.x + roi.y * srcSz.w + srcStp * blockIdx.x + srcStp * srcSz.d * blockIdx.y;
    out += dstStp * blockIdx.x + dstStp * srcSz.d * blockIdx.y;
        
    // gridDim.x - srcSz.d 
    // gridDim.y - srcSz.n
        
    unsigned int i = threadIdx.x; // blockDim.x <= roi.h
    while (i < roi.h){

        snFloat* pIn = in + i * srcSz.w,
               * pOut = out + i * roi.w;

        unsigned int j = threadIdx.y;  // blockDim.y <= roi.w
        while (j < roi.w){

            pOut[j] = pIn[j];

            j += blockDim.y;
        }

        i += blockDim.x;
    }  
}

__global__ void cropOut2In(roi roi, snSize srcSz, snFloat* in, snFloat* out){

    size_t srcStp = srcSz.w * srcSz.h,
           dstStp = roi.w * roi.h;

    in += roi.x + roi.y * srcSz.w + srcStp * blockIdx.x + srcStp * srcSz.d * blockIdx.y;
    out += dstStp * blockIdx.x + dstStp * srcSz.d * blockIdx.y;

    // gridDim.x - srcSz.d 
    // gridDim.y - srcSz.n

    unsigned int i = threadIdx.x;   // blockDim.x <= roi.h
    while (i < roi.h){

        snFloat* pIn = in + i * srcSz.w,
               * pOut = out + i * roi.w;

        unsigned int j = threadIdx.y;  // blockDim.y <= roi.w
        while (j < roi.w){

            pIn[j] = pOut[j];

            j += blockDim.y;
        }

        i += blockDim.x;
    }
}

void crop(bool inToOut, const roi& roi, const snSize& sz, snFloat* in, snFloat* out){

    dim3 dimBlock(16, 16);
    dim3 dimGrid(int(sz.d), int(sz.n));

    if (inToOut)
        cropIn2Out << < dimGrid, dimBlock >> >(roi, sz, in, out);
    else
        cropOut2In << < dimGrid, dimBlock >> >(roi, sz, in, out);
}
