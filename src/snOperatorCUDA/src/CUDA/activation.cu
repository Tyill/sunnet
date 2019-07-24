
#include <cuda_runtime.h>

#include "../stdafx.h"
#include "../structurs.h"
#include "../activationFunctions.h"


using namespace SN_Base;

__global__ void fv_sigmoid(snSize outsz, snFloat* output){
      
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
           outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){
               
        output[i] = 1.F / (1.F + exp(-output[i]));

        output[i] = (output[i] < 0.99999F) ? output[i] : 0.99999F;
        output[i] = (output[i] > 0.00001F) ? output[i] : 0.00001F;
      
        i += blockDim.x;
    }
}
__global__ void df_sigmoid(snSize outsz, snFloat* output, snFloat* grad){
        
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
           outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;
    grad += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        output[i] = output[i] * (1.F - output[i]);

        grad[i] *= output[i];

        i += blockDim.x;
    }
}

__global__ void fv_relu(snSize outsz, snFloat* output){
       
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        if (output[i] < 0)
           output[i] = 0;

        i += blockDim.x;
    }
};
__global__ void df_relu(snSize outsz, snFloat* output, snFloat* grad){
      
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;
    grad += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        output[i] = output[i] >= 0 ? 1.F : 0.F;

        grad[i] *= output[i];

        i += blockDim.x;
    }
};

__global__ void fv_leakyRelu(snSize outsz, snFloat* output){
       
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        output[i] = output[i] >= 0 ? output[i] : 0.01F * output[i];

        i += blockDim.x;
    }
}
__global__ void df_leakyRelu(snSize outsz, snFloat* output, snFloat* grad){
       
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;
    grad += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        output[i] = output[i] >= 0 ? 1 : 0.01F;

        grad[i] *= output[i];

        i += blockDim.x;
    }
}

__global__ void fv_elu(snSize outsz, snFloat* output){
        
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        output[i] = output[i] >= 0 ? output[i] : 0.01F * (exp(output[i]) - 1.F);

        i += blockDim.x;
    }
}
__global__ void df_elu(snSize outsz, snFloat* output, snFloat* grad){
       
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;
    grad += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        output[i] = output[i] >= 0 ? 1 : output[i] + 0.01F;

        grad[i] *= output[i];

        i += blockDim.x;
    }
}

void activationForward(const snSize& sz, snFloat* data, activeType active){
         
    dim3 dimBlock(128);
    dim3 dimGrid(int(sz.d), int(sz.n));
    
    switch (active){
       case activeType::sigmoid:   fv_sigmoid   <<< dimGrid, dimBlock >>>(sz, data); break;
       case activeType::relu:      fv_relu      <<< dimGrid, dimBlock >>>(sz, data); break;
       case activeType::leakyRelu: fv_leakyRelu <<< dimGrid, dimBlock >>>(sz, data); break;
       case activeType::elu:       fv_elu       <<< dimGrid, dimBlock >>>(sz, data); break;
       default: break;
    }    
}

void activationBackward(const snSize& sz, snFloat* data, snFloat* gradIn, activeType active){

    dim3 dimBlock(128);
    dim3 dimGrid(int(sz.d), int(sz.n));

    switch (active){
      case activeType::sigmoid:   df_sigmoid   <<< dimGrid, dimBlock >>>(sz, data, gradIn); break;
      case activeType::relu:      df_relu      <<< dimGrid, dimBlock >>>(sz, data, gradIn); break;
      case activeType::leakyRelu: df_leakyRelu <<< dimGrid, dimBlock >>>(sz, data, gradIn); break;
      case activeType::elu:       df_elu       <<< dimGrid, dimBlock >>>(sz, data, gradIn); break;
      default: break;
    }
}

