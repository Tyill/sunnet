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
#include "LossFunction.h"
#include "SNOperator/src/activeFunctions.h"

using namespace std;
using namespace SN_Base;


void LossFunction::load(std::map<std::string, std::string>& prms){

    baseOut_ = new Tensor();
    baseGrad_ = new Tensor();

    if (prms.find("lossType") != prms.end()){

        string stype = prms["lossType"];
        if (stype == "softMaxToCrossEntropy")
            lossType_ = lossType::softMaxACrossEntropy;
        else if (stype == "binaryCrossEntropy")
            lossType_ = lossType::binaryCrossEntropy;
        else
            statusMess("LossFunction::setInternPrm error: param 'lossType' = " + stype + " indefined");
    }
    else
        statusMess("LossFunction::setInternPrm error: not found param 'lossType'");
}

/// оператор - расчет ошибки
LossFunction::LossFunction(const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(name, node, prms){

    load(prms);
}


/// выполнить расчет
std::vector<std::string> LossFunction::Do(const learningParam& lernPrm, const std::vector<OperatorBase*>& neighbOpr){

    if (neighbOpr.size() == 1){
        if (lernPrm.action == snAction::forward)
            forward(neighbOpr[0]->getOutput());
        else
            backward(neighbOpr[0]->getGradient(), lernPrm);
    }
    else{
        if (lernPrm.action == snAction::forward){

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
                inBwTns_ +=  *neighbOpr[i]->getGradient();

            backward(&inBwTns_, lernPrm);
        }
    }

    return std::vector<std::string>();
}

void LossFunction::forward(Tensor* inTns){

    snSize tsz = inTns->size();
        
    // копируем себе
    baseOut_->setData(inTns->getData(), tsz);
    
    auto out = baseOut_->getData();

    switch (lossType_){
    case LossFunction::softMaxACrossEntropy:{

        if (auxParams_.find("sm_aux") == auxParams_.end())
            auxParams_["sm_aux"] = vector<snFloat>(tsz.w);

        // прогоняем через softMax каждое изобр отдельно
        snFloat* aux = auxParams_["sm_aux"].data();
        size_t nsz = tsz.n, width = tsz.w;
        for (size_t i = 0; i < nsz; ++i){

            snFloat maxv = *std::max_element(out, out + width);

            snFloat denom = 0.F;
            for (size_t j = 0; j < width; ++j){

                aux[j] = (out[j] - maxv > -20) ? std::exp(out[j] - maxv) : 0.1E-8F;
                denom += aux[j];
            }

            for (size_t j = 0; j < width; ++j)
                out[j] = aux[j] / denom;

            out += width;
        }
    }
    case LossFunction::binaryCrossEntropy:{        

        break;
    }
    break;
    default:
        statusMess("LossFunction::Do error: param 'lossType' indefined");
        break;
    }

}

void LossFunction::backward(Tensor* inTns, const learningParam& lernPrm){

    snSize tsz = inTns->size();
        
    // размер
    snSize grsz = baseGrad_->size();
    if (grsz != tsz)
        baseGrad_->resize(tsz);

    // градиент уже задан сверху? расчит ошибку не надо
    if (!lernPrm.isAutoCalcError){
        baseGrad_->setData(inTns->getData(), grsz);
        return;
    }

    auto smOut = baseOut_->getData();    // результат после forward
    auto target = inTns->getData();         // задан целевой результат
    auto grad = baseGrad_->getData();   // градиент ошибки на входе в lossFunc

    switch (lossType_){
    case LossFunction::softMaxACrossEntropy:{

        // считаем ошибку для всех изобр
        size_t nsz = tsz.n * tsz.w;
        for (size_t i = 0; i < nsz; ++i)
            grad[i] = smOut[i] - target[i];

        break;
    }

    case LossFunction::binaryCrossEntropy:{

        // считаем ошибку для всех изобр
        size_t nsz = tsz.n * tsz.w;
        for (size_t i = 0; i < nsz; ++i)
            grad[i] = (smOut[i] - target[i]) / (smOut[i] * (1.F - smOut[i]));
    
        break;
    }
    default:
        statusMess("LossFunction::Do error: param 'lossType' indefined");
        break;
    }
}
