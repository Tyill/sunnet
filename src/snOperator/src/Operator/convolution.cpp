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
#include "convolution.h"
#include "snAux/auxFunc.h"
#include "SNOperator/src/weightInit.h"
#include "SNOperator/src/activeFunctions.h"
#include "SNOperator/src/optimizer.h"
#include "SNOperator/src/structurs.h"
#include "SNOperator/src/mathFunctions.h"

using namespace std;
using namespace SN_Base;

/// сверточный слой

Convolution::Convolution(const string& name, const string& node, std::map<std::string, std::string>& prms) :
    OperatorBase(name, node, prms){
        
    load(prms);
}

void Convolution::load(std::map<std::string, std::string>& prms){

    baseOut_ = new Tensor();
    baseGrad_ = new Tensor();
    baseWeight_ = new Tensor();    

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
        
    setIntParam("kernel", false, true, kernel_);
    setIntParam("fWidth", false, false, fWidth_);
    setIntParam("fHeight", false, false, fHeight_);
        
    if ((prms.find("padding") != prms.end()) && (prms["padding"] == "-1"))
        isPaddingSame_ = true;
    else
        setIntParam("padding", true, false, paddingSet_);

    setIntParam("stride", true, false, stride_);
            
    // вспом массивы
    auxParams_["outGradExp"] = vector<snFloat>();
    auxParams_["dWeight"] = vector<snFloat>();
    auxParams_["dWPrev"] = vector<snFloat>();
    auxParams_["dWGrad"] = vector<snFloat>();    

    setInternPrm(prms);
}

bool Convolution::setInternPrm(std::map<std::string, std::string>& prms){

    basePrms_ = prms;

    if (prms.find("activeType") != prms.end()){

        string atype = prms["activeType"];
        if (atype == "none") activeType_ = activeType::none;
        else if (atype == "sigmoid") activeType_ = activeType::sigmoid;
        else if (atype == "relu") activeType_ = activeType::relu;
        else if (atype == "leakyRelu") activeType_ = activeType::leakyRelu;
        else if (atype == "elu") activeType_ = activeType::elu;
        else
            ERROR_MESS("param 'activeType' = " + atype + " indefined");
    }

    if (prms.find("optimizerType") != prms.end()){

        string optType = prms["optimizerType"];
        if (optType == "sgd") optimizerType_ = optimizerType::sgd;
        else if (optType == "sgdMoment") optimizerType_ = optimizerType::sgdMoment;
        else if (optType == "adagrad") optimizerType_ = optimizerType::adagrad;
        else if (optType == "adam") optimizerType_ = optimizerType::adam;
        else if (optType == "RMSprop") optimizerType_ = optimizerType::RMSprop;
        else
            ERROR_MESS("param 'optimizerType' = " + optType + " indefined");
    }

    if (prms.find("weightInitType") != prms.end()){

        string wInit = prms["weightInitType"];
        if (wInit == "uniform") weightInitType_ = weightInitType::uniform;
        else if (wInit == "he") weightInitType_ = weightInitType::he;
        else if (wInit == "lecun") weightInitType_ = weightInitType::lecun;
        else if (wInit == "xavier") weightInitType_ = weightInitType::xavier;
        else
            ERROR_MESS("param 'weightInitType' = " + wInit + " indefined");
    }

    if (prms.find("batchNormType") != prms.end()){

        string bnType = prms["batchNormType"];
        if (bnType == "none") batchNormType_ = batchNormType::none;
        else if (bnType == "beforeActive") batchNormType_ = batchNormType::beforeActive;
        else if (bnType == "postActive") batchNormType_ = batchNormType::postActive;
        else
            ERROR_MESS("param 'batchNormType' = " + bnType + " indefined");
    }

    if (prms.find("decayMomentDW") != prms.end())
        opt_decayMomentDW_ = stof(prms["decayMomentDW"]);

    if (prms.find("decayMomentWGr") != prms.end())
        opt_decayMomentWGr_ = stof(prms["decayMomentWGr"]);

    if (prms.find("lmbRegular") != prms.end())
        opt_lmbRegular_ = stof(prms["lmbRegular"]);

    if (prms.find("batchNormLr") != prms.end())
        for (auto& bn : bnPrm_)
            bn.lr = stof(prms["batchNormLr"]);
                
    return true;
}

std::vector<std::string> Convolution::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
       
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

void Convolution::forward(SN_Base::Tensor* inTns){
   
    snSize insz = inTns->size();

    /// размер вх данных изменился?
    if (insz != inSzMem_){
        inSzMem_ = insz;
        updateConfig(insz);
    }

    /// копируем со смещением padding для каждого изобр
    snFloat* pInTns = inTns->getData();
    snFloat* pDtMem = inDataExp_.data();
   
    if ((paddingW_ == 0) && (paddingH_ == 0))
        memcpy(pDtMem, pInTns, insz.size() * sizeof(snFloat));
    else{
        size_t paddW = paddingW_, paddH = paddingH_, sz = insz.h * insz.d * insz.n, stW = insz.w, stH = insz.h;
        pDtMem += (stW + paddW * 2) * paddH;
        for (size_t i = 0; i < sz; ++i){

            if ((i % stH == 0) && (i > 0))
                pDtMem += (stW + paddW * 2) * paddH * 2;

            pDtMem += paddW;
            for (size_t j = 0; j < stW; ++j)
                pDtMem[j] = pInTns[j];
            pDtMem += paddW + stW;

            pInTns += stW;
        }
    }
    
    /// расчет выходных значений нейронов
    snFloat* out = baseOut_->getData(), *weight = baseWeight_->getData();
    snSize outsz = baseOut_->size();
    fwdConvolution(kernel_, fWidth_, fHeight_, stride_, weight, inDataExpSz_, inDataExp_.data(), outsz, out);

    /// batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        batchNorm(true, outsz, out);

    /// функция активации
    switch (activeType_){
    case activeType::sigmoid:   fv_sigmoid(out, outsz.size()); break;
    case activeType::relu:      fv_relu(out, outsz.size()); break;
    case activeType::leakyRelu: fv_leakyRelu(out, outsz.size()); break;
    case activeType::elu:       fv_elu(out, outsz.size()); break;
    default: break;
    }

    /// batchNorm
    if (batchNormType_ == batchNormType::postActive)
        batchNorm(true, outsz, out);
}

void Convolution::backward(SN_Base::Tensor* inTns, const operationParam& operPrm){
    
    snFloat* gradIn = inTns->getData();

    /// batchNorm
    if (batchNormType_ == batchNormType::postActive)
        batchNorm(false, inTns->size(), gradIn);

    // проходим через ф-ю активации, если есть
    if (activeType_ != activeType::none){

        snFloat* out = baseOut_->getData();
        
        // производная функции активации
        size_t osz = baseOut_->size().size();
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
        batchNorm(false, inTns->size(), gradIn);

    // расчет вых градиента и коррекции весов
    bool isSame = (paddingW_ == 0) && (paddingH_ == 0);
    snFloat* pGrOutExp = isSame ? baseGrad_->getData() : auxParams_["outGradExp"].data();
   
    snFloat* weight = baseWeight_->getData();
    snFloat* dWeight = auxParams_["dWeight"].data();
    bwdConvolution(kernel_, fWidth_, fHeight_, stride_, weight, inDataExpSz_, inDataExp_.data(),
        baseOut_->size(), gradIn, pGrOutExp, dWeight);
        
    if (!isSame){
        /// копируем градиент со смещением padding для каждого изобр
        snFloat* pGrOut = baseGrad_->getData();
        size_t paddW = paddingW_, paddH = paddingH_, sz = inSzMem_.h * inSzMem_.d * inSzMem_.n, stW = inSzMem_.w, stH = inSzMem_.h;
        pGrOutExp += (stW + paddW * 2) * paddH;
        for (size_t i = 0; i < sz; ++i){

            if ((i % stH == 0) && (i > 0))
                pGrOutExp += (stW + paddW * 2) * paddH * 2;

            pGrOutExp += paddW;
            for (size_t j = 0; j < stW; ++j)
                pGrOut[j] = pGrOutExp[j];
            pGrOutExp += paddW + stW;

            pGrOut += stW;
        }
    }

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

void Convolution::batchNorm(bool fwBw, const SN_Base::snSize& outsz, snFloat* out){

    snFloat* share = (snFloat*)calloc(outsz.w * outsz.h * outsz.n, sizeof(snFloat));
    for (size_t i = 0; i < outsz.d; ++i){

        snFloat* pSh = share;
        snFloat* pOut = out + (outsz.w * outsz.h) * i;
        for (size_t j = 0; j < outsz.n; ++j){

            memcpy(pSh, pOut, outsz.w * outsz.h * sizeof(snFloat));

            pSh += outsz.w * outsz.h;
            pOut += outsz.w * outsz.h * outsz.d;
        }

        fwdBatchNorm(outsz, share, share, bnPrm_[i]);
    }
}

void Convolution::updateConfig(const snSize& newsz){
    
    size_t stp = fWidth_ * fHeight_ * newsz.d, ntp = (stp + 1) * kernel_;
        
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
        
    snSize outSz(0, 0, kernel_, newsz.n);
          
    if (isPaddingSame_){
        outSz.w = newsz.w;
        outSz.h = newsz.h;

        paddingW_ = (newsz.w * (stride_ - 1) + fWidth_ - stride_) / 2;
        paddingH_ = (newsz.h * (stride_ - 1) + fHeight_ - stride_) / 2;
    }
    else{
        paddingW_ = paddingH_ = paddingSet_;

        outSz.w = (newsz.w + paddingW_ * 2 - fWidth_) / stride_ + 1;
        outSz.h = (newsz.h + paddingH_ * 2 - fHeight_) / stride_ + 1;
    }

    // проверка коррект
    int res = (newsz.w + paddingW_ * 2 - fWidth_) % stride_;
    if (res != 0)
        ERROR_MESS("not correct param 'stride' or 'fWidth'");

    res = (newsz.h + paddingH_ * 2 - fHeight_) % stride_;
    if (res != 0)
        ERROR_MESS("not correct param 'stride' or 'fHeight'");


    inDataExpSz_ = snSize(newsz.w + paddingW_ * 2, newsz.h + paddingH_ * 2, newsz.d, newsz.n);
    inDataExp_.resize(inDataExpSz_.size());

    memset(inDataExp_.data(), 0, inDataExpSz_.size() * sizeof(snFloat));

    baseOut_->resize(outSz);
    baseGrad_->resize(newsz);
        
    // вспом массивы
    if (inDataExpSz_ != newsz)
        auxParams_["outGradExp"].resize(inDataExpSz_.size(), 0);
    auxParams_["dWeight"].resize(ntp, 0);
    auxParams_["dWPrev"].resize(ntp, 0);
    auxParams_["dWGrad"].resize(ntp, 0);

    size_t osz = outSz.w * outSz.h;

    for (size_t i = 0; i < outSz.d; ++i){ //!!!!!!!!!!!!
        /*string nm = "bn_norm" + to_string(i);
        auxParams_[nm] = vector<snFloat>(osz * newsz.n, 0);  bnPrm_[i].norm = auxParams_[nm].data();
        
        nm = "bn_mean" + to_string(i);
        auxParams_[nm] = vector<snFloat>(osz, 0);            bnPrm_[i].mean = auxParams_[nm].data();
                                                             
        nm = "bn_varce" + to_string(i);                      
        auxParams_[nm] = vector<snFloat>(osz, 0);            bnPrm_[i].varce = auxParams_[nm].data();
                                                             
        nm = "bn_scale" + to_string(i);                      
        auxParams_[nm] = vector<snFloat>(osz, 1.F);          bnPrm_[i].scale = auxParams_[nm].data();
                                                             
        nm = "bn_dScale" + to_string(i);                     
        auxParams_[nm] = vector<snFloat>(osz, 0);            bnPrm_[i].dScale = auxParams_[nm].data();
                                                             
        nm = "bn_schift" + to_string(i);                     
        auxParams_[nm] = vector<snFloat>(osz, 0);            bnPrm_[i].schift = auxParams_[nm].data();
                                                             
        nm = "bn_dSchift" + to_string(i);                    
        auxParams_[nm] = vector<snFloat>(osz, 0);            bnPrm_[i].dSchift = auxParams_[nm].data();
                                                             
        nm = "bn_onc" + to_string(i);                        
        auxParams_[nm].resize(newsz.n, 1.F);                 bnPrm_[i].onc = auxParams_[nm].data();*/
    }
} 



