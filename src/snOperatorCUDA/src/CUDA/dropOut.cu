
#include <cuda_runtime.h>
#include <curand.h>

#include "../stdafx.h"

using namespace SN_Base;

__global__ void dropOutLern(SN_Base::snFloat dropOut, const snSize& outsz, snFloat* rnd, SN_Base::snFloat* out){
    
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
           outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    out += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    
    if (rnd[i] < dropOut){

        while (i < outStepByD){

            out[i] = 0.F;

            i += blockDim.x;
        }
    }
}

__global__ void dropOutInf(SN_Base::snFloat dropOut, const snSize& outsz, snFloat* out){

    size_t outStepByD = outsz.w * outsz.h,     // step out by input
           outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    out += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;

    while (i < outStepByD){

        out[i] *= (1.F - dropOut);

        i += blockDim.x;
    }
}


void dropOut(bool isLern, snFloat dropOut, const snSize& outsz, snFloat* inout){
       
    if (isLern){
        int blockSz = 128;

        float* rndData = nullptr;
        cuAssert(cudaMalloc((void**)&rndData, blockSz * sizeof(float)));

        curandGenerator_t gen;
        cuAssert(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

        cuAssert(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

        cuAssert(curandGenerateUniform(gen, rndData, blockSz));

        dim3 dimBlock(blockSz);
        dim3 dimGrid(int(outsz.d), int(outsz.n));
               
        dropOutLern << < dimGrid, dimBlock >> >(dropOut, outsz, rndData, inout);
        
        cuAssert(curandDestroyGenerator(gen));
        cuAssert(cudaFree(rndData));
    }
    else{
     
        dim3 dimBlock(128);
        dim3 dimGrid(int(outsz.d), int(outsz.n));
        
        dropOutInf << <dimGrid, dimBlock >> >(dropOut, outsz, inout);
    }
}