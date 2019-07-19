
#include <cuda_runtime.h>
#include <cudnn.h>

#include "../stdafx.h"

using namespace SN_Base;


/// tensor - input data and output data of each node of the network.

Tensor::Tensor(const snSize& sz) : sz_(sz){

    size_t ssz = sz.size();

    if (ssz > 0){               
        cuAssert(cudaMalloc(&dataGPU_, ssz * sizeof(snFloat)));
        cuAssert(cudaMemset(dataGPU_, 0, ssz * sizeof(snFloat)));        
    }
}

Tensor::~Tensor(){
    if (dataGPU_)
        cuAssert(cudaFree(dataGPU_));

    if (dataCPU_)
        free(dataCPU_);
}

Tensor::Tensor(const Tensor& other){

    setDataGPU(other.getDataGPU(), other.size());
}
      
Tensor& Tensor::operator=(const Tensor& other){

    setDataGPU(other.getDataGPU(), other.size());

    return *this;
}

Tensor& Tensor::operator+=(const Tensor& other){

    ASSERT_MESS(other == *this, "");
       
   
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other){

    ASSERT_MESS(other == *this, "");

    
    return *this;
}

void Tensor::setDataGPU(const snFloat* data, const snSize& nsz){

    size_t nnsz = nsz.size();
    ASSERT_MESS(data && (nnsz > 0), "");

    if (sz_.size() < nnsz){
     
        if (dataGPU_)
            cuAssert(cudaFree(dataGPU_));
 
        cuAssert(cudaMalloc(&dataGPU_, nnsz * sizeof(snFloat)));
    }

    cuAssert(cudaMemcpy(dataGPU_, data, nnsz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    sz_ = nsz;
}

void Tensor::setDataCPU(const snFloat* data, const snSize& nsz){

    size_t nnsz = nsz.size();
    ASSERT_MESS(data && (nnsz > 0), "");

    if (sz_.size() < nnsz){

        if (dataGPU_)
            cuAssert(cudaFree(dataGPU_));

        cuAssert(cudaMalloc(&dataGPU_, nnsz * sizeof(snFloat)));
    }

    cuAssert(cudaMemcpy(dataGPU_, data, nnsz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyHostToDevice));
    sz_ = nsz;
}

snFloat* Tensor::getDataGPU() const{

    return dataGPU_;
}

snFloat* Tensor::getDataCPU() const{
      
    size_t csz = sz_.size();

    dataCPU_ = (snFloat*)realloc(dataCPU_, csz * sizeof(snFloat));

    cuAssert(cudaMemcpy(dataCPU_, dataGPU_, csz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    return dataCPU_;
}

void Tensor::resize(const snSize& nsz){

    size_t nnsz = nsz.size(), csz = sz_.size();
    ASSERT_MESS(nnsz > 0, "");

    if (csz < nnsz){

        snFloat* mem = nullptr;
        cuAssert(cudaMalloc(&mem, nnsz * sizeof(snFloat)));

        if (dataGPU_){
            if (csz > 0)
               cuAssert(cudaMemcpy(mem, dataGPU_, csz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
            cuAssert(cudaFree(dataGPU_));
        }
        dataGPU_ = mem;

        cuAssert(cudaMemset(dataGPU_ + csz, 0, (nnsz - csz) * sizeof(snFloat)));
    }

    sz_ = nsz;
}