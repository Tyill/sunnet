
#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "../stdafx.h"

using namespace SN_Base;

#ifndef cuCHECK
#define cuCHECK(func) if (func != 0){ std::cout << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl; return;}
#endif

/// tensor - input data and output data of each node of the network.

Tensor::Tensor(const snSize& sz) : sz_(sz){

    size_t ssz = sz.size();

    if (ssz > 0){
        cuCHECK(cudaMalloc(&data_, ssz * sizeof(snFloat)));
        cuCHECK(cudaMemset(data_, 0, ssz * sizeof(snFloat)));
    }
}

Tensor::~Tensor(){
    if (data_)
        cuCHECK(cudaFree(data_)); 
}

Tensor::Tensor(const Tensor& other){
    setData(other.getData(), other.size());
}
      
Tensor& Tensor::operator=(const Tensor& other){

    setData(other.getData(), other.size());

    return *this;
}

Tensor& Tensor::operator+=(const Tensor& other){

    assert(other == *this);

    auto od = other.getData();

    //size_t sz = this->size().size();
    //for (size_t i = 0; i < sz; ++i){
    //    data_[i] += od[i];
    //}

    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other){

    assert(other == *this);

    auto od = other.getData();

   /* size_t sz = this->size().size();
    for (size_t i = 0; i < sz; ++i){
        data_[i] -= od[i];
    }*/

    return *this;
}

void Tensor::setData(const snFloat* data, const snSize& nsz){

    size_t nnsz = nsz.size();
    assert(data && (nnsz > 0));

    if (sz_.size() < nnsz){
     
        if (data_)
            cuCHECK(cudaFree(data_));
 
        cuCHECK(cudaMalloc(&data_, nnsz * sizeof(snFloat)));
    }

    cuCHECK(cudaMemcpy(data_, data, nnsz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    sz_ = nsz;
}

void Tensor::setDataCPU2GPU(const snFloat* data, const snSize& nsz, const size_t& offset){

    assert(sz_.size() == (nsz.size() + offset));

    cuCHECK(cudaMemcpy(data_ + offset, data, nsz.size() * sizeof(snFloat), cudaMemcpyHostToDevice));
}

snFloat* Tensor::getData() const{

    return data_;
}

void Tensor::getDataGPU2CPU(snFloat* out, const snSize& osz) const{
    
    assert(sz_ == osz);

    cuCHECK(cudaMemcpy(out, data_, sz_.size() * sizeof(snFloat), cudaMemcpyDeviceToHost));
}

void Tensor::resize(const snSize& nsz){

    size_t nnsz = nsz.size(), csz = sz_.size();
    assert(nnsz > 0);

    if (csz < nnsz){

        snFloat* mem = nullptr;
        cuCHECK(cudaMalloc(&mem, nnsz * sizeof(snFloat)));

        if (data_){
            cuCHECK(cudaMemcpy(mem, data_, csz * sizeof(snFloat), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
            cuCHECK(cudaFree(data_));
        }
        data_ = mem;

        cuCHECK(cudaMemset(data_ + csz, 0, (nnsz - csz) * sizeof(snFloat)));
    }

    sz_ = nsz;
}

void Tensor::tfree(){
    if (data_)
        cuCHECK(cudaFree(data_));

    data_ = nullptr;
    sz_ = snSize(0, 0, 0, 0, 0);
}
