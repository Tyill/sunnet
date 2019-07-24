
#include <cuda_runtime.h>

#include "../stdafx.h"

using namespace SN_Base;



__global__ void summInf(SN_Base::snSize sz, SN_Base::snFloat* one, const SN_Base::snFloat* two){

    size_t outStepByD = sz.w * sz.h,        // step out by input
        outStepByN = outStepByD * sz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    one += blockIdx.x * outStepByD + blockIdx.y * outStepByN;
    two += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;

    while (i < outStepByD){

        one[i] += two[i];

        i += blockDim.x;
    }
}

__global__ void differenceInf(SN_Base::snSize sz, SN_Base::snFloat* one, const SN_Base::snFloat* two){

    size_t outStepByD = sz.w * sz.h,        // step out by input
        outStepByN = outStepByD * sz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    one += blockIdx.x * outStepByD + blockIdx.y * outStepByN;
    two += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;

    while (i < outStepByD){

        one[i] -= two[i];

        i += blockDim.x;
    }
}

__global__ void meanInf(SN_Base::snSize sz, SN_Base::snFloat* one, const SN_Base::snFloat* two){

    size_t outStepByD = sz.w * sz.h,        // step out by input
        outStepByN = outStepByD * sz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    one += blockIdx.x * outStepByD + blockIdx.y * outStepByN;
    two += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;

    while (i < outStepByD){

        one[i] = (one[i] + two[i]) / 2;

        i += blockDim.x;
    }
}

void summ(const SN_Base::snSize& sz, SN_Base::snFloat* inout, const SN_Base::snFloat* two){

    dim3 dimBlock(128);
    dim3 dimGrid(int(sz.d), int(sz.n));

    summInf << < dimGrid, dimBlock >> >(sz, inout, two);
}

void difference(const SN_Base::snSize& sz, SN_Base::snFloat* inout, const SN_Base::snFloat* two){

    dim3 dimBlock(128);
    dim3 dimGrid(int(sz.d), int(sz.n));

    differenceInf << < dimGrid, dimBlock >> >(sz, inout, two);
}

void mean(const SN_Base::snSize& sz, SN_Base::snFloat* inout, const SN_Base::snFloat* two){

    dim3 dimBlock(128);
    dim3 dimGrid(int(sz.d), int(sz.n));

    meanInf << < dimGrid, dimBlock >> >(sz, inout, two);
}

