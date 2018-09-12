//
// SkyNet Project
// Copyright (C) 2018 by Contributors <https://github.com/Tyill/skynet>
//
// This code is licensed under the MIT License.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
#include "../stdafx.h"
#include "pooling.h"
#include "snAux/auxFunc.h"
#include "SNOperator/src/structurs.h"

using namespace std;
using namespace SN_Base;

/// объединяющий слой

Pooling::Pooling(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
    OperatorBase(net, name, node, prms){
        
    load(prms);
}

void Pooling::load(std::map<std::string, std::string>& prms){
    
    baseOut_ = new Tensor();
    baseGrad_ = new Tensor();
    
    auto setIntParam = [&prms, this](const string& name, bool isZero, bool checkExist, size_t& value){

        if ((prms.find(name) != prms.end()) && SN_Aux::is_number(prms[name])){

            size_t v = stoi(prms[name]);
            if ((v > 0) || (isZero && (v == 0)))
                value = v;
            else
                ERROR_MESS("param '" + name + (isZero ? "' < 0" : "' <= 0"));
        }
        else if (checkExist)
            ERROR_MESS("not found (or not numder) param '" + name + "'");
    };
    
    setIntParam("kernel", false, false, kernel_);
    
    if (prms.find("pool") != prms.end()){

        string atype = prms["pool"];
        if (atype == "max") poolType_ = poolType::max;
        else if (atype == "avg") poolType_ = poolType::avg;
        else
            ERROR_MESS("param 'pool' = " + atype + " indefined");
    }

    if (prms.find("mode") != prms.end()){

        string mode = prms["mode"];
        if (mode == "CPU") calcMode_ = calcMode::CPU;
        else if (mode == "CUDA") calcMode_ = calcMode::CUDA;
        else if (mode == "OpenCL") calcMode_ = calcMode::OpenCL;
        else
            ERROR_MESS("param 'mode' = " + mode + " indefined");
    }

    basePrms_ = prms;
}

std::vector<std::string> Pooling::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
    
    if (neighbOpr.size() > 1){
        ERROR_MESS("neighbOpr.size() > 1");
        return std::vector < std::string > {"noWay"};
    }

    if (operPrm.action == snAction::forward)
        forward(neighbOpr[0]->getOutput());
    else
        backward(neighbOpr[0]->getGradient(), operPrm);
    
    return std::vector<std::string>();
}

void Pooling::forward(SN_Base::Tensor* inTns){

    snSize insz = inTns->size();

    /// размер вх данных изменился?
    if (insz != inSzMem_){
        inSzMem_ = insz;
        updateConfig(insz);
    }

    /// копируем со смещением padding для каждого изобр
    snFloat* pDtMem = inTns->getData();
    if (isPadding_){
        pDtMem = inDataExp_.data();
        paddingOffs(false, insz, pDtMem, inTns->getData());
        insz = inDataExpSz_;
    }

    /// расчет выходных значений
    snFloat* out = baseOut_->getData();
   
    switch (calcMode_){
    case calcMode::CPU:    forwardCPU(poolType_, kernel_, insz, pDtMem, baseOut_->size(), out, outInx_.data()); break;
    case calcMode::CUDA:   forwardCUDA(poolType_, kernel_, insz, pDtMem, baseOut_->size(), out, outInx_.data(), gpuParams_); break;
    case calcMode::OpenCL: forwardOCL(poolType_, kernel_, insz, pDtMem, baseOut_->size(), out, outInx_.data(), gpuParams_); break;
    }       
}

void Pooling::backward(SN_Base::Tensor* inTns, const operationParam& operPrm){

    snFloat* gradIn = inTns->getData();
    
    snFloat* pGrOutExp = !isPadding_ ? baseGrad_->getData() : auxParams_["outGradExp"].data();

    /// расчет вых градиента
    switch (calcMode_){
    case calcMode::CPU:    backwardCPU(poolType_, kernel_, baseOut_->size(), outInx_.data(), gradIn, inDataExpSz_, pGrOutExp); break;
    case calcMode::CUDA:   backwardCUDA(poolType_, kernel_, baseOut_->size(), outInx_.data(), gradIn, inDataExpSz_, pGrOutExp, gpuParams_); break;
    case calcMode::OpenCL: backwardOCL(poolType_, kernel_, baseOut_->size(), outInx_.data(), gradIn, inDataExpSz_, pGrOutExp, gpuParams_); break;
    }
   
    if (isPadding_)
        paddingOffs(true, inSzMem_, pGrOutExp, baseGrad_->getData());
}

void Pooling::paddingOffs(bool in2out, const SN_Base::snSize& insz, SN_Base::snFloat* in, SN_Base::snFloat* out){
    
    /// копируем со смещением padding для каждого изобр

    size_t paddW = paddingW_, paddH = paddingH_,
        sz = insz.h * insz.d * insz.n, stW = insz.w, stH = insz.h;
    if (in2out){
        in += (stW + paddW * 2) * paddH;
        for (size_t i = 0; i < sz; ++i){

            if ((i % stH == 0) && (i > 0))
                in += (stW + paddW * 2) * paddH * 2;

            in += paddW;
            for (size_t j = 0; j < stW; ++j)
                out[j] = in[j];
            in += paddW + stW;

            out += stW;
        }
    }
    else{
        in += (stW + paddW * 2) * paddH;
        for (size_t i = 0; i < sz; ++i){

            if ((i % stH == 0) && (i > 0))
                in += (stW + paddW * 2) * paddH * 2;

            in += paddW;
            for (size_t j = 0; j < stW; ++j)
                in[j] = out[j];
            in += paddW + stW;

            out += stW;
        }
    }

}

void Pooling::updateConfig(const snSize& newsz){
           
    snSize outSz(0, 0, newsz.d, newsz.n);
                 
    outSz.w = (newsz.w - kernel_) / kernel_ + 1;
    outSz.h = (newsz.h - kernel_) / kernel_ + 1;

    // проверка коррект
    size_t resW = (newsz.w - kernel_) % kernel_, resH = (newsz.h - kernel_) % kernel_;    
    isPadding_ = (resW != 0) || (resH != 0);

    inDataExpSz_ = newsz;

    if (isPadding_){   
      
        paddingW_ = 1;
        paddingH_ = 1;

        outSz.w = (newsz.w + paddingW_ * 2 - kernel_) / kernel_ + 1;
        outSz.h = (newsz.h + paddingH_ * 2 - kernel_) / kernel_ + 1;

        inDataExpSz_ = snSize(newsz.w + paddingW_ * 2, newsz.h + paddingH_ * 2, newsz.d, newsz.n);
        inDataExp_.resize(inDataExpSz_.size());

        memset(inDataExp_.data(), 0, inDataExpSz_.size() * sizeof(snFloat));
    }
        
    baseOut_->resize(outSz);
    baseGrad_->resize(newsz);

    outInx_.resize(outSz.size(), 0);

    if (isPadding_)
        auxParams_["outGradExp"].resize(inDataExpSz_.size(), 0);

    if (calcMode_ == calcMode::CUDA)
        iniParamCUDA(inDataExpSz_, outSz, kernel_, gpuParams_);
    else if (calcMode_ == calcMode::OpenCL)
        iniParamOCL(inDataExpSz_, outSz, kernel_, gpuParams_);
}