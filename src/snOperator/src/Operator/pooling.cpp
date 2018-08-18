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
    
    if (prms.find("poolType") != prms.end()){

        string atype = prms["poolType"];
        if (atype == "max") poolType_ = poolType::max;
        else if (atype == "avg") poolType_ = poolType::avg;
        else
            ERROR_MESS("param 'poolType' = " + atype + " indefined");
    }

    basePrms_ = prms;
}

std::vector<std::string> Pooling::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
    
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

void Pooling::forward(SN_Base::Tensor* inTns){

    snSize insz = inTns->size();

    /// размер вх данных изменился?
    if (insz != inSzMem_){
        inSzMem_ = insz;
        updateConfig(insz);
    }
            
    /// расчет выходных значений
    snFloat* out = baseOut_->getData();
    fwdPooling((int)poolType_, kernel_, insz, inTns->getData(), baseOut_->size(), out, outInx_.data());
       
}

void Pooling::backward(SN_Base::Tensor* inTns, const operationParam& operPrm){

    snFloat* gradIn = inTns->getData(), *gradOut = baseGrad_->getData();
    
    /// расчет вых градиента
    bwdPooling((int)poolType_, kernel_, baseOut_->size(), outInx_.data(), gradIn, inSzMem_, gradOut);
}

void Pooling::updateConfig(const snSize& newsz){
           
    snSize outSz(0, 0, newsz.d, newsz.n);
                 
    outSz.w = (newsz.w - kernel_) / kernel_ + 1;
    outSz.h = (newsz.h - kernel_) / kernel_ + 1;

    // проверка коррект
    size_t resW = (newsz.w - kernel_) % kernel_, resH = (newsz.h - kernel_) % kernel_;
    if ((resW != 0) || (resH != 0))
        ERROR_MESS("not correct param 'kernel'");
       
    baseOut_->resize(outSz);
    baseGrad_->resize(newsz);

    outInx_.resize(outSz.size(), 0);
}