
#include <cuda_runtime.h>
#include <cudnn.h>

#include "../stdafx.h"

using namespace SN_Base;

void dropOutForward(SN_Base::snFloat dropOut, SN_Base::snFloat* inout, const SN_Base::snSize& iosz, uint32_t deviceId){
       
    cudaSetDevice(deviceId);

            
}

void dropOutBackward(SN_Base::snFloat* inout, const SN_Base::snSize& iosz, uint32_t deviceId){

    cudaSetDevice(deviceId);
 
   

}