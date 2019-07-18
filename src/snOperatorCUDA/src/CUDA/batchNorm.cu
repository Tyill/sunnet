
#include <cuda_runtime.h>
#include <cudnn.h>

#include "../stdafx.h"
#include "../structurs.h"

using namespace SN_Base;


void batchNormInit(SN_Base::snFloat* inout, const SN_Base::snSize& iosz, void** pGpuPrm){
        
    bool isFirst = false;

    gpuParams* gpuPrm = (gpuParams*)*pGpuPrm;

    if (!gpuPrm){

        gpuPrm = new gpuParams();
        memset(gpuPrm, 0, sizeof(gpuParams));
        *pGpuPrm = gpuPrm;

        cudnnHandle_t cudnn = nullptr;
        cuAssert(cudnnCreate(&cudnn));
        gpuPrm->cudnn = cudnn;      

        isFirst = true;
    }

    if (gpuPrm->inszMem != iosz){
             
        cudnnActivationMode_t actMode;

        switch (atype){
            case activeType::sigmoid:   actMode = cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID; break;
            case activeType::relu:      actMode = cudnnActivationMode_t::CUDNN_ACTIVATION_RELU; break;
            case activeType::leakyRelu: actMode = cudnnActivationMode_t::CUDNN_ACTIVATION_CLIPPED_RELU; break;
            case activeType::elu:       actMode = cudnnActivationMode_t::CUDNN_ACTIVATION_ELU; break;
            default:                    actMode = cudnnActivationMode_t::CUDNN_ACTIVATION_RELU; break;
        }
        
        // activ_desc
        cudnnActivationDescriptor_t activ_desc = nullptr;
        cuAssert(cudnnCreateActivationDescriptor(&activ_desc));
        cuAssert(cudnnSetActivationDescriptor(activ_desc, actMode, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN, 0.0));
        if (!isFirst)
            cuAssert(cudnnDestroyActivationDescriptor(gpuPrm->activ_desc));
        gpuPrm->activ_desc = activ_desc;

        // input
        cudnnTensorDescriptor_t x_desc = nullptr;
        cuAssert(cudnnCreateTensorDescriptor(&x_desc));
        cuAssert(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, int(iosz.n), int(iosz.d), int(iosz.h), int(iosz.w)));
        if (!isFirst)
            cuAssert(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm->x_desc));
        gpuPrm->x_desc = x_desc;

        // dinput
        cudnnTensorDescriptor_t dx_desc = nullptr;
        cuAssert(cudnnCreateTensorDescriptor(&dx_desc));
        cuAssert(cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, int(iosz.n), int(iosz.d), int(iosz.h), int(iosz.w)));
        if (!isFirst)
            cuAssert(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm->dx_desc));
        gpuPrm->dx_desc = dx_desc;

        // output
        cudnnTensorDescriptor_t y_desc;
        cuAssert(cudnnCreateTensorDescriptor(&y_desc));
        cuAssert(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, int(iosz.n), int(iosz.d), int(iosz.h), int(iosz.w)));
        if (!isFirst)
            cuAssert(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm->y_desc));
        gpuPrm->y_desc = y_desc;

        // doutput
        cudnnTensorDescriptor_t dy_desc;
        cuAssert(cudnnCreateTensorDescriptor(&dy_desc));
        cuAssert(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, int(iosz.n), int(iosz.d), int(iosz.h), int(iosz.w)));
        if (!isFirst)
            cuAssert(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)gpuPrm->dy_desc));
        gpuPrm->dy_desc = dy_desc;
       
        gpuPrm->inszMem = iosz;
    }
}

void batchNormFree(uint32_t deviceId, void* pGpuPrm){

    cudaSetDevice(deviceId);

    gpuParams* gpuPrm = (gpuParams*)pGpuPrm;

    if (!gpuPrm) return;

    cuAssert(cudnnDestroy(gpuPrm->cudnn));
    cuAssert(cudnnDestroyActivationDescriptor(gpuPrm->activ_desc));
    cuAssert(cudnnDestroyTensorDescriptor(gpuPrm->x_desc));
    cuAssert(cudnnDestroyTensorDescriptor(gpuPrm->y_desc));
}

void batchNormForward(activeType atype, SN_Base::snFloat* inout, const SN_Base::snSize& iosz, uint32_t deviceId, void** pGpuPrm){
       
    cudaSetDevice(deviceId);

    activationInit(atype, inout, iosz, pGpuPrm);

    gpuParams* gpuPrm = (gpuParams*)*pGpuPrm;
      
    cuAssert(cudaMemcpy(gpuPrm->in_mem, inout, iosz.size() * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    float alpha = 1.0, beta = 0.0;

    cuAssert(cudnnActivationForward(gpuPrm->cudnn,
        gpuPrm->activ_desc,
        &alpha,
        gpuPrm->x_desc,
        gpuPrm->in_mem,
        &beta,
        gpuPrm->y_desc,
        inout));        
}

void batchNormBackward(SN_Base::snFloat* inprev, SN_Base::snFloat* outprev, SN_Base::snFloat* inout, const SN_Base::snSize& iosz, uint32_t deviceId, void* pGpuPrm){

    cudaSetDevice(deviceId);
    
    gpuParams* gpuPrm = (gpuParams*)pGpuPrm;

    cuAssert(cudaMemcpy(gpuPrm->in_mem, inout, iosz.size() * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    float alpha = 1.0, beta = 0.0;

    cuAssert(cudnnActivationBackward(gpuPrm->cudnn,
                                     gpuPrm->activ_desc,
                                     &alpha,
                                     gpuPrm->y_desc,
                                     outprev,
                                     gpuPrm->dy_desc,
                                     gpuPrm->in_mem,
                                     gpuPrm->x_desc,
                                     inprev,
                                     &beta,
                                     gpuPrm->dx_desc,
                                     inout));
}