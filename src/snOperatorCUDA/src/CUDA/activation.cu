
#include <cuda_runtime.h>

#include "../stdafx.h"
#include "../structurs.h"
#include "../activationFunctions.h"

using namespace SN_Base;

__global__ void fv_sigmoid(const snSize& outsz, snFloat* output){
      
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
           outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){
               
        output[i] = 1.F / (1.F + std::exp(-output[i]));

        output[i] = (output[i] < 0.99999F) ? output[i] : 0.99999F;
        output[i] = (output[i] > 0.00001F) ? output[i] : 0.00001F;
      
        i += blockDim.x;
    }
}
__global__ void df_sigmoid(const snSize& outsz, snFloat* output){
        
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
           outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        output[i] = output[i] * (1.F - output[i]);

        i += blockDim.x;
    }
}

__global__ void fv_relu(const snSize& outsz, snFloat* output){
       
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        output[i] = output[i] >= 0 ? output[i] : 0;

        i += blockDim.x;
    }
};
__global__ void df_relu(const snSize& outsz, snFloat* output){
      
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        output[i] = output[i] >= 0 ? 1.F : 0.F;

        i += blockDim.x;
    }
};

__global__ void fv_leakyRelu(const snSize& outsz, snFloat* output){
       
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
__global__ void df_leakyRelu(const snSize& outsz, snFloat* output){
       
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        output[i] = output[i] >= 0 ? 1 : 0.01F;

        i += blockDim.x;
    }
}

__global__ void fv_elu(const snSize& outsz, snFloat* output){
        
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
__global__ void df_elu(const snSize& outsz, snFloat* output){
       
    size_t outStepByD = outsz.w * outsz.h,     // step out by input
        outStepByN = outStepByD * outsz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    output += blockIdx.x * outStepByD + blockIdx.y * outStepByN;

    unsigned int i = threadIdx.x;
    while (i < outStepByD){

        output[i] = output[i] >= 0 ? 1 : output[i] + 0.01F;

        i += blockDim.x;
    }
}

void activationForward(const snSize& sz, snFloat* data, activeType active, uint32_t deviceId){
       
    cudaSetDevice(deviceId);
      
    switch (active){
    case activeType::sigmoid:   fv_sigmoid(sz, data); break;
       case activeType::relu:      fv_relu(data, sz); break;
       case activeType::leakyRelu: fv_leakyRelu(data, sz); break;
       case activeType::elu:       fv_elu(data, sz); break;
       default: break;
    }
}

void activeFuncBackward(const snSize& sz, snFloat* data, activeType active, uint32_t deviceId){

    cudaSetDevice(deviceId);
    
    switch (active){
      case activeType::sigmoid:   df_sigmoid(data, sz); break;
      case activeType::relu:      df_relu(data, sz); break;
      case activeType::leakyRelu: df_leakyRelu(data, sz); break;
      case activeType::elu:       df_elu(data, sz); break;
      default: break;
    }
}

