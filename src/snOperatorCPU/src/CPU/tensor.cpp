
#include "snBase/snBase.h"

using namespace SN_Base;

/// tensor - input data and output data of each node of the network.

Tensor::Tensor(const snSize& sz) : sz_(sz){

    size_t ssz = sz.size();

    if (ssz > 0)
        data_ = (snFloat*)calloc(ssz, sizeof(snFloat));
}

Tensor::~Tensor(){
    if (data_) free(data_);
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

    size_t sz = this->size().size();
    for (size_t i = 0; i < sz; ++i){
        data_[i] += od[i];
    }

    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other){

    assert(other == *this);

    auto od = other.getData();

    size_t sz = this->size().size();
    for (size_t i = 0; i < sz; ++i){
        data_[i] -= od[i];
    }

    return *this;
}

void Tensor::setData(const snFloat* data, const snSize& nsz){

    size_t nnsz = nsz.size();
    assert(data && (nnsz > 0));

    if (sz_.size() < nnsz)
        data_ = (snFloat*)realloc(data_, nnsz * sizeof(snFloat));

    memcpy(data_, data, nnsz * sizeof(snFloat));
    sz_ = nsz;
}

snFloat* Tensor::getData() const{

    return data_;
}

void Tensor::getDataForCPU(snFloat* out, const snSize& osz) const{

    assert(sz_ == osz);

    memcpy(out, data_, sz_.size() * sizeof(snFloat));
}

void Tensor::resize(const snSize& nsz){

    size_t nnsz = nsz.size(), csz = sz_.size();
    assert(nnsz > 0);

    if (csz < nnsz){
        data_ = (snFloat*)realloc(data_, nnsz * sizeof(snFloat));
        memset(data_ + csz, 0, (nnsz - csz) * sizeof(snFloat));
    }

    sz_ = nsz;
}

void Tensor::tfree(){
    if (data_) free(data_);
    data_ = nullptr;
    sz_ = snSize(0, 0, 0, 0, 0);
}
