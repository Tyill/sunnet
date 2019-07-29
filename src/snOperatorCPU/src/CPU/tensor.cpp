
#include "snBase/snBase.h"
#include <cstring>

using namespace SN_Base;

/// tensor - input data and output data of each node of the network.

Tensor::Tensor(const snSize& sz) : sz_(sz){

    size_t ssz = sz.size();

    if (ssz > 0)
        dataCPU_ = (snFloat*)calloc(ssz, sizeof(snFloat));
}

Tensor::~Tensor(){

    if (dataCPU_)
        free(dataCPU_);
}

Tensor::Tensor(const Tensor& other){
    setDataCPU(other.getDataCPU(), other.size());
}
      
Tensor& Tensor::operator=(const Tensor& other){

    setDataCPU(other.getDataCPU(), other.size());

    return *this;
}

Tensor& Tensor::operator+=(const Tensor& other){

    ASSERT_MESS(other == *this, "");

    auto od = other.getDataCPU();

    size_t sz = this->size().size();
    for (size_t i = 0; i < sz; ++i){
        dataCPU_[i] += od[i];
    }

    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other){

    ASSERT_MESS(other == *this, "");

    auto od = other.getDataCPU();

    size_t sz = this->size().size();
    for (size_t i = 0; i < sz; ++i){
        dataCPU_[i] -= od[i];
    }

    return *this;
}

void Tensor::setDataCPU(const snFloat* data, const snSize& nsz){

    size_t nnsz = nsz.size();
    ASSERT_MESS(data && (nnsz > 0), "");

    if (sz_.size() < nnsz)
        dataCPU_ = (snFloat*)realloc(dataCPU_, nnsz * sizeof(snFloat));

    memcpy(dataCPU_, data, nnsz * sizeof(snFloat));
    sz_ = nsz;
}

snFloat* Tensor::getDataCPU() const{

    return dataCPU_;
}

void Tensor::resize(const snSize& nsz){

    size_t nnsz = nsz.size(), csz = sz_.size();
    ASSERT_MESS(nnsz > 0, "");

    if (csz < nnsz){
        dataCPU_ = (snFloat*)realloc(dataCPU_, nnsz * sizeof(snFloat));
        memset(dataCPU_ + csz, 0, (nnsz - csz) * sizeof(snFloat));
    }

    sz_ = nsz;
}