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
#include "FullyConnected.h"
#include "snAux/auxFunc.h"
#include "SNOperator/src/weightInit.h"
#include "SNOperator/src/activeFunctions.h"
#include "SNOperator/src/optimizer.h"
#include "SNOperator/src/structurs.h"
#include "SNOperator/src/mathFunctions.h"

using namespace std;
using namespace SN_Base;

/// полносвязный слой

FullyConnected::FullyConnected(const string& name, const string& node, std::map<std::string, std::string>& prms) :
    OperatorBase(name, node, prms){
        
    load(prms);
}

void FullyConnected::load(std::map<std::string, std::string>& prms){
    
    baseOut_ = new Tensor();
    baseGrad_ = new Tensor();
    baseWeight_ = new Tensor();    
    
    if ((prms.find("kernel") != prms.end()) && SN_Aux::is_number(prms["kernel"])){

        size_t kernel = stoi(prms["kernel"]);
        if (kernel > 0)
            kernel_ = kernel;
        else
            statusMess("FullyConnected::setInternPrm error: param 'kernel' <= 0");
    }
    else
        statusMess("FullyConnected::setInternPrm error: not found (or not numder) param 'kernel'");

    baseOut_->resize(snSize(kernel_));

    baseBatchNorm_.mean.resize(kernel_);                  bnPrm_.mean = baseBatchNorm_.mean.data();
    baseBatchNorm_.varce.resize(kernel_);                 bnPrm_.varce = baseBatchNorm_.varce.data();
    baseBatchNorm_.scale.resize(kernel_, 1);              bnPrm_.scale = baseBatchNorm_.scale.data();
    baseBatchNorm_.schift.resize(kernel_);                bnPrm_.schift = baseBatchNorm_.schift.data();
    baseBatchNorm_.sz.w = kernel_;

    // вспом массивы
    auxParams_["dWeight"] = vector<snFloat>();
    auxParams_["dWPrev"] = vector<snFloat>();
    auxParams_["dWGrad"] = vector<snFloat>();
    auxParams_["bn_norm"] = vector<snFloat>();               bnPrm_.norm = auxParams_["bn_norm"].data();
    auxParams_["bn_dScale"] = vector<snFloat>(kernel_, 0);   bnPrm_.dScale = auxParams_["bn_dScale"].data();
    auxParams_["bn_dSchift"] = vector<snFloat>(kernel_, 0);  bnPrm_.dSchift = auxParams_["bn_dSchift"].data();
    auxParams_["bn_onc"] = vector<snFloat>();                 bnPrm_.onc = auxParams_["bn_onc"].data();

    setInternPrm(prms);
}

bool FullyConnected::setInternPrm(std::map<std::string, std::string>& prms){

    basePrms_ = prms;

    if (prms.find("activeType") != prms.end()){

        string atype = prms["activeType"];
        if (atype == "none") activeType_ = activeType::none;
        else if (atype == "sigmoid") activeType_ = activeType::sigmoid;
        else if (atype == "relu") activeType_ = activeType::relu;
        else if (atype == "leakyRelu") activeType_ = activeType::leakyRelu;
        else if (atype == "elu") activeType_ = activeType::elu;
        else
            statusMess("FullyConnected::setInternPrm error: param 'activeType' = " + atype + " indefined");
    }

    if (prms.find("optimizerType") != prms.end()){

        string optType = prms["optimizerType"];
        if (optType == "sgd") optimizerType_ = optimizerType::sgd;
        else if (optType == "sgdMoment") optimizerType_ = optimizerType::sgdMoment;
        else if (optType == "adagrad") optimizerType_ = optimizerType::adagrad;
        else if (optType == "adam") optimizerType_ = optimizerType::adam;
        else if (optType == "RMSprop") optimizerType_ = optimizerType::RMSprop;
        else
            statusMess("FullyConnected::setInternPrm error: param 'optimizerType' = " + optType + " indefined");
    }

    if (prms.find("weightInitType") != prms.end()){

        string wInit = prms["weightInitType"];
        if (wInit == "uniform") weightInitType_ = weightInitType::uniform;
        else if (wInit == "he") weightInitType_ = weightInitType::he;
        else if (wInit == "lecun") weightInitType_ = weightInitType::lecun;
        else if (wInit == "xavier") weightInitType_ = weightInitType::xavier;
        else
            statusMess("FullyConnected::setInternPrm error: param 'weightInitType' = " + wInit + " indefined");
    }

    if (prms.find("batchNormType") != prms.end()){

        string bnType = prms["batchNormType"];
        if (bnType == "none") batchNormType_ = batchNormType::none;
        else if (bnType == "beforeActive") batchNormType_ = batchNormType::beforeActive;
        else if (bnType == "postActive") batchNormType_ = batchNormType::postActive;
        else
            statusMess("FullyConnected::setInternPrm error: param 'batchNormType' = " + bnType + " indefined");
    }

    if (prms.find("decayMomentDW") != prms.end())
        opt_decayMomentDW_ = stof(prms["decayMomentDW"]);

    if (prms.find("decayMomentWGr") != prms.end())
        opt_decayMomentWGr_ = stof(prms["decayMomentWGr"]);

    if (prms.find("lmbRegular") != prms.end())
        opt_lmbRegular_ = stof(prms["lmbRegular"]);

    if (prms.find("batchNormLr") != prms.end())
        bnPrm_.lr = stof(prms["batchNormLr"]);
    
    return true;
}

/// выполнить расчет
std::vector<std::string> FullyConnected::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){

    if (neighbOpr.size() == 1){
        if (operPrm.action == snAction::forward)
            forward(neighbOpr[0]->getOutput());
        else
            backward(neighbOpr[0]->getGradient(), operPrm);
    }
    else{
        if (operPrm.action == snAction::forward){

            inFwTns_ = *neighbOpr[0]->getOutput();

            size_t sz = neighbOpr.size();
            for (size_t i = 1; i < sz; ++i)
                inFwTns_ += *neighbOpr[i]->getOutput();

            forward(&inFwTns_);
        }
        else{

            inBwTns_ = *neighbOpr[0]->getGradient();

            size_t sz = neighbOpr.size();
            for (size_t i = 1; i < sz; ++i)
                inBwTns_ += *neighbOpr[i]->getGradient();

            backward(&inBwTns_, operPrm);
        }
    }
    
    return std::vector<std::string>();
}

void FullyConnected::forward(SN_Base::Tensor* inTns){

    snSize insz = inTns->size();

    /// размер вх данных изменился?
    if (insz != inSzMem_){
        inSzMem_ = insz;
        updateConfig(insz);
    }

    /// копируем со смещением X0 для каждого изобр
    snFloat* pInTns = inTns->getData();
    snFloat* pDtMem = inDataExp_.data();
    size_t stp = insz.w * insz.h * insz.d, ssz = stp * sizeof(snFloat);
    for (size_t i = 0; i < insz.n; ++i){
        memcpy(pDtMem + i * stp + i + 1, pInTns + i * stp, ssz);
    }
    
    /// расчет выходных значений нейронов
    snFloat* out = baseOut_->getData();
    fwdFullyConnected(kernel_, insz, pDtMem, baseWeight_->getData(), out);

    /// batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        fwdBatchNorm(baseOut_->size(), out, out, bnPrm_);

    /// функция активации
    switch (activeType_){
    case activeType::sigmoid:   fv_sigmoid(out, kernel_ * insz.n); break;
    case activeType::relu:      fv_relu(out, kernel_ * insz.n); break;
    case activeType::leakyRelu: fv_leakyRelu(out, kernel_ * insz.n); break;
    case activeType::elu:       fv_elu(out, kernel_ * insz.n); break;
    default: break;
    }

    /// batchNorm
    if (batchNormType_ == batchNormType::postActive)
        fwdBatchNorm(baseOut_->size(), out, out, bnPrm_);

}

void FullyConnected::backward(SN_Base::Tensor* inTns, const operationParam& operPrm){

    snFloat* gradIn = inTns->getData();

    /// batchNorm
    if (batchNormType_ == batchNormType::postActive)
        bwdBatchNorm(inTns->size(), gradIn, gradIn, bnPrm_);

    // проходим через ф-ю активации, если есть
    if (activeType_ != activeType::none){

        snFloat* out = baseOut_->getData();
        
        // производная функции активации
        size_t osz = kernel_ * inSzMem_.n;
        switch (activeType_){
        case activeType::sigmoid:   df_sigmoid(out, osz); break;
        case activeType::relu:      df_relu(out, osz); break;
        case activeType::leakyRelu: df_leakyRelu(out, osz); break;
        case activeType::elu:       df_elu(out, osz); break;
        default: break;
        }

        // обновл градиент
        for (size_t i = 0; i < osz; ++i) gradIn[i] *= out[i];
    }

    /// batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        bwdBatchNorm(inTns->size(), gradIn, gradIn, bnPrm_);

    // расчет вых градиента и коррекции весов
    snFloat* gradOut = baseGrad_->getData();
    snFloat* weight = baseWeight_->getData();
    snFloat* dWeight = auxParams_["dWeight"].data();
    bwdFullyConnected(kernel_, weight, inSzMem_, inDataExp_.data(), gradIn, gradOut, dWeight);
        
    // корректируем веса
    snFloat* dWPrev = auxParams_["dWPrev"].data();
    snFloat* dWGrad = auxParams_["dWGrad"].data();
    size_t wsz = baseWeight_->size().size();
    
    switch (optimizerType_){
    case optimizerType::sgd:       opt_sgd(dWeight, weight, wsz, operPrm.lr, opt_lmbRegular_); break;
    case optimizerType::sgdMoment: opt_sgdMoment(dWeight, dWPrev, weight, wsz, operPrm.lr, opt_lmbRegular_, opt_decayMomentDW_); break;
    case optimizerType::RMSprop:   opt_RMSprop(dWeight, dWGrad, weight, wsz, operPrm.lr, opt_lmbRegular_, opt_decayMomentWGr_); break;
    case optimizerType::adagrad:   opt_adagrad(dWeight, dWGrad, weight, wsz, operPrm.lr, opt_lmbRegular_); break;
    case optimizerType::adam:      opt_adam(dWeight, dWPrev, dWGrad, weight, wsz, operPrm.lr, opt_lmbRegular_, opt_decayMomentDW_, opt_decayMomentWGr_); break;
    default: break;
    }

}

void FullyConnected::updateConfig(const snSize& newsz){
    
    size_t stp = newsz.w * newsz.h * newsz.d, ntp = (stp + 1) * kernel_;

    inDataExp_.resize((stp + 1) * newsz.n);
    snFloat* pDtMem = inDataExp_.data();
    for (size_t i = 0; i < newsz.n; ++i){
        pDtMem[i * stp + i] = 1.0F;
    }

    // имеющиеся веса оставляем как есть, остаток инициализируем
    size_t wcsz = baseWeight_->size().size();
    if (ntp > wcsz){
                
        baseWeight_->resize(snSize(kernel_, stp + 1));
        snFloat* wd = baseWeight_->getData();
        switch (weightInitType_){
        case weightInitType::uniform: wi_uniform(wd + wcsz, ntp - wcsz); break;
        case weightInitType::he: wi_he(wd + wcsz, ntp - wcsz, stp + 1); break;
        case weightInitType::lecun:wi_lecun(wd + wcsz, ntp - wcsz, kernel_); break;
        case weightInitType::xavier:wi_xavier(wd + wcsz, ntp - wcsz, stp + 1, kernel_); break;
        }
    }
    
    baseOut_->resize(snSize(kernel_, 1, 1, newsz.n));
    baseGrad_->resize(newsz);
        
    // вспом массивы
    auxParams_["dWeight"].resize(ntp, 0);
    auxParams_["dWPrev"].resize(ntp, 0);
    auxParams_["dWGrad"].resize(ntp, 0);
    auxParams_["bn_norm"].resize(newsz.n * kernel_, 0); bnPrm_.norm = auxParams_["bn_norm"].data();
    auxParams_["bn_onc"].resize(newsz.n, 1.F);          bnPrm_.onc = auxParams_["bn_onc"].data();    
    
} 



