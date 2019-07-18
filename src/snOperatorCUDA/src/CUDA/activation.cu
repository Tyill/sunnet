
#include <cuda_runtime.h>
#include <cudnn.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../stdafx.h"
#include "../structurs.h"
#include "../activeFunctions.h"

using namespace SN_Base;


void activationForward(size_t sz, SN_Base::snFloat* data, activeType active, uint32_t deviceId){
       
    cudaSetDevice(deviceId);

    switch (active){
       case activeType::sigmoid:   fv_sigmoid(data, sz); break;
       case activeType::relu:      fv_relu(data, sz); break;
       case activeType::leakyRelu: fv_leakyRelu(data, sz); break;
       case activeType::elu:       fv_elu(data, sz); break;
       default: break;
    }
}

void activeFuncBackward(size_t sz, SN_Base::snFloat* data, activeType active, uint32_t deviceId){

    cudaSetDevice(deviceId);
    
    switch (active){
      case activeType::sigmoid:   df_sigmoid(data, sz); break;
      case activeType::relu:      df_relu(data, sz); break;
      case activeType::leakyRelu: df_leakyRelu(data, sz); break;
      case activeType::elu:       df_elu(data, sz); break;
      default: break;
    }
}

void fv_sigmoid(snFloat* ioVal, size_t sz){

    for (size_t i = 0; i < sz; ++i){

        ioVal[i] = 1.F / (1.F + std::exp(-ioVal[i]));

        ioVal[i] = (ioVal[i] < 0.99999F) ? ioVal[i] : 0.99999F;
        ioVal[i] = (ioVal[i] > 0.00001F) ? ioVal[i] : 0.00001F;
    }
}
void df_sigmoid(snFloat* ioSigm, size_t sz){

    for (size_t i = 0; i < sz; ++i){

        ioSigm[i] = ioSigm[i] * (1.F - ioSigm[i]);
    }
}

void fv_relu(snFloat* ioVal, size_t sz){

    for (size_t i = 0; i < sz; ++i){

        ioVal[i] = ioVal[i] >= 0 ? ioVal[i] : 0;
    }
};
void df_relu(snFloat* ioRelu, size_t sz){

    for (size_t i = 0; i < sz; ++i){

        ioRelu[i] = ioRelu[i] >= 0 ? 1.F : 0.F;
    }
};

void fv_leakyRelu(snFloat* ioVal, size_t sz, snFloat minv){

    for (size_t i = 0; i < sz; ++i){

        ioVal[i] = ioVal[i] >= 0 ? ioVal[i] : minv * ioVal[i];
    }
}
void df_leakyRelu(snFloat* ioRelu, size_t sz, snFloat minv){

    for (size_t i = 0; i < sz; ++i){

        ioRelu[i] = ioRelu[i] >= 0 ? 1 : minv;
    }
}

void fv_elu(snFloat* ioVal, size_t sz, snFloat minv){

    for (size_t i = 0; i < sz; ++i){

        ioVal[i] = ioVal[i] >= 0 ? ioVal[i] : minv * (exp(ioVal[i]) - 1.F);
    }
}
void df_elu(snFloat* ioElu, size_t sz, snFloat minv){

    for (size_t i = 0; i < sz; ++i){

        ioElu[i] = ioElu[i] >= 0 ? 1 : ioElu[i] + minv;
    }
}
