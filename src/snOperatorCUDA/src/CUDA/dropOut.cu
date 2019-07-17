
#include <cuda_runtime.h>
#include <cudnn.h>

#include "../stdafx.h"

using namespace SN_Base;

struct gpuParams{

    cudnnHandle_t cudnn = 0;
    cudnnDropoutDescriptor_t drop_desc = 0;
    cudnnTensorDescriptor_t in_desc = 0;
    cudnnTensorDescriptor_t out_desc = 0;
   
    size_t wsSz = 0;
    size_t state_sizes = 0;
    SN_Base::snSize inszMem = 0;

    snFloat* in_mem = nullptr;

    void* state_memory = 0;   
    void* d_ws = 0;
};

void dropOutInit(void** pGpuPrm, SN_Base::snFloat dropOut, SN_Base::snFloat* inout, const SN_Base::snSize& outsz){
        
    bool isFirst = false;

    gpuParams* gpuPrm = (gpuParams*)*pGpuPrm;

    if (!gpuPrm){

        gpuPrm = new gpuParams();
        memset(gpuPrm, 0, sizeof(gpuParams));
        *pGpuPrm = gpuPrm;

        cudnnHandle_t cudnn = nullptr;
        cuAssert(cudnnCreate(&cudnn));
        gpuPrm->cudnn = cudnn;

        cuAssert(cudnnDropoutGetStatesSize(gpuPrm->cudnn, &gpuPrm->state_sizes));
        cuAssert(cudaMalloc(&gpuPrm->state_memory, gpuPrm->state_sizes));

        isFirst = true;
    }

    if (gpuPrm->inszMem != outsz){

        // drop_desc
        cudnnDropoutDescriptor_t drop_desc = nullptr;
        cuAssert(cudnnCreateDropoutDescriptor(&drop_desc));
        cuAssert(cudnnSetDropoutDescriptor(drop_desc, gpuPrm->cudnn, dropOut, gpuPrm->state_memory, gpuPrm->state_sizes, 1234ULL));
        if (!isFirst)
            cuAssert(cudnnDestroyDropoutDescriptor(gpuPrm->drop_desc));
        gpuPrm->drop_desc = drop_desc;

        // input
        cudnnTensorDescriptor_t in_desc = nullptr;
        cuAssert(cudnnCreateTensorDescriptor(&in_desc));
        cuAssert(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, int(outsz.n), int(outsz.d), int(outsz.h), int(outsz.w)));
        if (!isFirst)
            cuAssert(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm->in_desc));
        gpuPrm->in_desc = in_desc;

        // output
        cudnnTensorDescriptor_t out_desc;
        cuAssert(cudnnCreateTensorDescriptor(&out_desc));
        cuAssert(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, int(outsz.n), int(outsz.d), int(outsz.h), int(outsz.w)));
        if (!isFirst)
            cuAssert(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm->out_desc));
        gpuPrm->out_desc = out_desc;

        // ws
        cuAssert(cudnnDropoutGetReserveSpaceSize(in_desc, &gpuPrm->wsSz));
        cuAssert(cudaMalloc(&gpuPrm->d_ws, gpuPrm->wsSz));

        // in mem
        if (!isFirst)
            cuAssert(cudaFree(gpuPrm->in_mem));
        cuAssert(cudaMalloc(&gpuPrm->in_mem, outsz.size() * sizeof(snFloat)));

        gpuPrm->inszMem = outsz;
    }
}

void dropOutFree(uint32_t deviceId, void* pGpuPrm){

    cudaSetDevice(deviceId);

    gpuParams* gpuPrm = (gpuParams*)pGpuPrm;

    if (!gpuPrm) return;

    cuAssert(cudnnDestroy(gpuPrm->cudnn));
    cuAssert(cudnnDestroyDropoutDescriptor(gpuPrm->drop_desc));
    cuAssert(cudnnDestroyTensorDescriptor(gpuPrm->in_desc));
    cuAssert(cudnnDestroyTensorDescriptor(gpuPrm->out_desc));

    cuAssert(cudaFree(gpuPrm->d_ws));
    cuAssert(cudaFree(gpuPrm->state_memory));
}

void dropOutForward(SN_Base::snFloat dropOut, SN_Base::snFloat* inout, const SN_Base::snSize& outsz, uint32_t deviceId, void** pGpuPrm){
       
    cudaSetDevice(deviceId);

    dropOutInit(pGpuPrm, dropOut, inout, outsz);

    gpuParams* gpuPrm = (gpuParams*)*pGpuPrm;
  
    cuAssert(cudaMemcpy(gpuPrm->in_mem, inout, outsz.size() * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    
    cuAssert(cudnnDropoutForward(gpuPrm->cudnn,
        gpuPrm->drop_desc,
        gpuPrm->in_desc,
        gpuPrm->in_mem,
        gpuPrm->out_desc,
        inout,
        gpuPrm->d_ws,
        gpuPrm->wsSz));
        
}

void dropOutBackward(SN_Base::snFloat dropOut, SN_Base::snFloat* inout, const SN_Base::snSize& outsz, uint32_t deviceId, void** pGpuPrm){

    cudaSetDevice(deviceId);

    dropOutInit(pGpuPrm, dropOut, inout, outsz);

    gpuParams* gpuPrm = (gpuParams*)*pGpuPrm;

    cuAssert(cudaMemcpy(gpuPrm->in_mem, inout, outsz.size() * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    cuAssert(cudnnDropoutBackward(gpuPrm->cudnn,
        gpuPrm->drop_desc,
        gpuPrm->in_desc,
        gpuPrm->in_mem,
        gpuPrm->out_desc,
        inout,
        gpuPrm->d_ws,
        gpuPrm->wsSz));

}