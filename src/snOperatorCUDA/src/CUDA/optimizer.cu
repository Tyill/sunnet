
#include <cuda_runtime.h>

#include "../stdafx.h"
#include "../structurs.h"

using namespace SN_Base;

/// adaptive gradient method
__global__ void opt_adagrad(snFloat* dW, snFloat* ioWGr, snFloat* ioW, const snSize& sz, snFloat alpha, snFloat lambda, snFloat eps){

    size_t wStepByD = sz.w * sz.h,     
           wStepByN = wStepByD * sz.d;      
                   
    ioWGr += blockIdx.x * wStepByD + blockIdx.y * wStepByN;
    ioW += blockIdx.x * wStepByD + blockIdx.y * wStepByN;
    dW += blockIdx.x * wStepByD + blockIdx.y * wStepByN;

    unsigned int i = threadIdx.x;
    while (i < wStepByD){

        ioWGr[i] += dW[i] * dW[i];
        ioW[i] -= alpha * (dW[i] + ioW[i] * lambda) / (sqrt(ioWGr[i]) + eps);

        i += blockDim.x;
    }    
}

/// RMSprop
__global__ void opt_RMSprop(snFloat* dW, snFloat* ioWGr, snFloat* ioW, const snSize& sz, snFloat alpha, snFloat lambda, snFloat mu, snFloat eps){

    size_t wStepByD = sz.w * sz.h,
           wStepByN = wStepByD * sz.d;

    ioWGr += blockIdx.x * wStepByD + blockIdx.y * wStepByN;
    ioW += blockIdx.x * wStepByD + blockIdx.y * wStepByN;
    dW += blockIdx.x * wStepByD + blockIdx.y * wStepByN;

    unsigned int i = threadIdx.x;
    while (i < wStepByD){

        ioWGr[i] = ioWGr[i] * mu + (1.F - mu) * dW[i] * dW[i];
        ioW[i] -= alpha * (dW[i] + ioW[i] * lambda) / std::sqrt(ioWGr[i] + eps);

        i += blockDim.x;
    }   
}

/// adam
__global__ void opt_adam(snFloat* dW, snFloat* iodWPrev, snFloat* ioWGr, snFloat* ioW, const snSize& sz, snFloat alpha, snFloat lambda, snFloat mudW, snFloat muGr, snFloat eps){

    size_t wStepByD = sz.w * sz.h,
           wStepByN = wStepByD * sz.d;

    iodWPrev += blockIdx.x * wStepByD + blockIdx.y * wStepByN;
    ioWGr += blockIdx.x * wStepByD + blockIdx.y * wStepByN;
    ioW += blockIdx.x * wStepByD + blockIdx.y * wStepByN;
    dW += blockIdx.x * wStepByD + blockIdx.y * wStepByN;

    unsigned int i = threadIdx.x;
    while (i < wStepByD){

        iodWPrev[i] = iodWPrev[i] * mudW - (1.F - mudW) * alpha * (dW[i] + ioW[i] * lambda);

        ioWGr[i] = ioWGr[i] * muGr + (1.F - muGr) * dW[i] * dW[i];

        ioW[i] += iodWPrev[i] / std::sqrt(ioWGr[i] + eps);

        i += blockDim.x;
    }
    
}

/// SGD without momentum
__global__ void opt_sgd(snFloat* dW, snFloat* ioW, const snSize& sz, snFloat alpha, snFloat lambda){
    
    size_t wStepByD = sz.w * sz.h,
           wStepByN = wStepByD * sz.d;

    ioW += blockIdx.x * wStepByD + blockIdx.y * wStepByN;
    dW += blockIdx.x * wStepByD + blockIdx.y * wStepByN;

    unsigned int i = threadIdx.x;
    while (i < wStepByD){

        ioW[i] -= alpha * (dW[i] + lambda * ioW[i]);

        i += blockDim.x;
    }    
}

/// SGD with momentum
__global__ void opt_sgdMoment(snFloat* dW, snFloat* iodWPrev, snFloat* ioW, const snSize& sz, snFloat alpha, snFloat lambda, snFloat mu){

    size_t wStepByD = sz.w * sz.h,
           wStepByN = wStepByD * sz.d;

    iodWPrev += blockIdx.x * wStepByD + blockIdx.y * wStepByN;
    ioW += blockIdx.x * wStepByD + blockIdx.y * wStepByN;
    dW += blockIdx.x * wStepByD + blockIdx.y * wStepByN;

    unsigned int i = threadIdx.x;
    while (i < wStepByD){

        iodWPrev[i] = mu * iodWPrev[i] - alpha * (dW[i] + ioW[i] * lambda);
        ioW[i] += iodWPrev[i];

        i += blockDim.x;
    }
}

void optimizer(snFloat* dWeight, snFloat* dWPrev, snFloat* dWGrad, snFloat* weight, const SN_Base::snSize& wsz, snFloat alpha, snFloat lambda, snFloat mudW, snFloat muGr, optimizerType otype){

    dim3 dimBlock(128);
    dim3 dimGrid(int(wsz.d), int(wsz.n));

    switch (otype){
      case optimizerType::sgd:       opt_sgd       <<<dimGrid, dimBlock>>> (dWeight, weight, wsz, alpha, lambda); break;
      case optimizerType::sgdMoment: opt_sgdMoment <<<dimGrid, dimBlock>>> (dWeight, dWPrev, weight, wsz, alpha, lambda, mudW); break;
      case optimizerType::RMSprop:   opt_RMSprop   <<<dimGrid, dimBlock>>> (dWeight, dWGrad, weight, wsz, alpha, lambda, muGr, 1e-8F); break;
      case optimizerType::adagrad:   opt_adagrad   <<<dimGrid, dimBlock>>> (dWeight, dWGrad, weight, wsz, alpha, lambda, 1e-8F); break;
      case optimizerType::adam:      opt_adam      <<<dimGrid, dimBlock>>> (dWeight, dWPrev, dWGrad, weight, wsz, alpha, lambda, mudW, muGr, 1e-8F); break;
      default: break;
    }
}
