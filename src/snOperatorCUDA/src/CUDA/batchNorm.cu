
#include <cuda_runtime.h>
#include <cudnn.h>

#include "../stdafx.h"
#include "../structurs.h"

using namespace SN_Base;


void batchNormForward(activeType atype, SN_Base::snFloat* inout, const SN_Base::snSize& iosz, uint32_t deviceId, void** pGpuPrm){
       
    cudaSetDevice(deviceId);

   
}

void batchNormBackward(SN_Base::snFloat* inprev, SN_Base::snFloat* outprev, SN_Base::snFloat* inout, const SN_Base::snSize& iosz, uint32_t deviceId, void* pGpuPrm){

    cudaSetDevice(deviceId);
    
   
}