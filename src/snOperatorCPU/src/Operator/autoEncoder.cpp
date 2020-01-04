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
#include "snAux/auxFunc.h"
#include "snOperatorCPU/src/Operator/autoEncoder.h"

using namespace std;
using namespace SN_Base;

/// AE layer
AutoEncoder::AutoEncoder(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
                         OperatorBase(net, name, node, prms){

    betweenFC_ = new FullyConnected(net, name, node, prms);
    outFC_ = new FullyConnected(net, name, node, prms);
       
}

AutoEncoder::~AutoEncoder(){
    
    delete betweenFC_;
    delete outFC_;
}

bool AutoEncoder::setInternPrm(std::map<std::string, std::string>& prms){

    basePrms_ = prms;
       
    return betweenFC_->setInternPrm(prms) && outFC_->setInternPrm(prms);
}

bool AutoEncoder::setBatchNorm(const batchNorm& bn){

    size_t osz = bn.sz.size();

    /*auxParams_["bn_mean"] = vector<snFloat>(osz, 0);     baseBatchNorm_.mean = auxParams_["bn_mean"].data();
    auxParams_["bn_varce"] = vector<snFloat>(osz, 1);    baseBatchNorm_.varce = auxParams_["bn_varce"].data();
    auxParams_["bn_scale"] = vector<snFloat>(osz, 1);    baseBatchNorm_.scale = auxParams_["bn_scale"].data();
    auxParams_["bn_schift"] = vector<snFloat>(osz, 0);   baseBatchNorm_.schift = auxParams_["bn_schift"].data();

    memcpy(baseBatchNorm_.mean, bn.mean, osz * sizeof(snFloat));
    memcpy(baseBatchNorm_.varce, bn.varce, osz * sizeof(snFloat));
    memcpy(baseBatchNorm_.scale, bn.scale, osz * sizeof(snFloat));
    memcpy(baseBatchNorm_.schift, bn.schift, osz * sizeof(snFloat));
    */
    baseBatchNorm_.sz = bn.sz;

    return true;
}

batchNorm AutoEncoder::getBatchNorm() const{

    batchNorm betwBN = ((OperatorBase*)betweenFC_)->getBatchNorm();
    batchNorm outBN = ((OperatorBase*)outFC_)->getBatchNorm();

    batchNorm bn;

    size_t osz = betwBN.sz.size();
        
    auxParams_["bn_mean"] = vector<snFloat>(osz, 0);     bn.mean = auxParams_["bn_mean"].data();
    auxParams_["bn_varce"] = vector<snFloat>(osz, 1);    bn.varce = auxParams_["bn_varce"].data();
    auxParams_["bn_scale"] = vector<snFloat>(osz, 1);    bn.scale = auxParams_["bn_scale"].data();
    auxParams_["bn_schift"] = vector<snFloat>(osz, 0);   bn.schift = auxParams_["bn_schift"].data();

    memcpy(bn.mean, betwBN.mean, osz * sizeof(snFloat));
    memcpy(bn.varce, betwBN.varce, osz * sizeof(snFloat));
    memcpy(bn.scale, betwBN.scale, osz * sizeof(snFloat));
    memcpy(bn.schift, betwBN.schift, osz * sizeof(snFloat));

    bn.sz = snSize(osz);

    return bn;
}

bool AutoEncoder::setWeight(const snFloat* data, const snSize& dsz){

    ((OperatorBase*)betweenFC_)->setWeight(data, snSize(dsz.w, dsz.h));

    ((OperatorBase*)outFC_)->setWeight(data + dsz.w * dsz.h, snSize(dsz.w, dsz.h));
};

const SN_Base::Tensor& AutoEncoder::getWeight() const{

    const Tensor& betwW = ((OperatorBase*)betweenFC_)->getWeight();
    const Tensor& outW = ((OperatorBase*)outFC_)->getWeight();

    snSize osz = outW.size();

    weight_.resize(snSize(osz.w, osz.h, 2));

    memcpy(weight_.getDataCPU(), betwW.getDataCPU(), osz.w * osz.h * sizeof(snFloat));

    memcpy(weight_.getDataCPU() + osz.w * osz.h, outW.getDataCPU(), osz.w * osz.h * sizeof(snFloat));

    return weight_;
}

const SN_Base::Tensor& AutoEncoder::getOutput() const{

    const Tensor& tOut = ((OperatorBase*)outFC_)->getOutput();

    if (tOut.size() != inSzMem_)
      outFC_->resizeOut(inSzMem_);

    return tOut;
}

const SN_Base::Tensor& AutoEncoder::getGradient() const{

    return ((OperatorBase*)betweenFC_)->getGradient();
}

std::vector<std::string> AutoEncoder::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
    
    if (operPrm.action == snAction::forward){

        if (neighbOpr.size() > 1){
            ERROR_MESS("neighbOpr.size() > 1");
            return std::vector < std::string > {"noWay"};
        }

        const Tensor& inTns = neighbOpr[0]->getOutput();

        snSize insz = inTns.size();

        if (inSzMem_ != insz){
            inSzMem_ = insz;

            outFC_->setUnits(insz.w * insz.h * insz.d);
        }

        betweenFC_->Do(operPrm, neighbOpr);

        outFC_->Do(operPrm, vector<OperatorBase*> { (OperatorBase*)betweenFC_ });     
    }
    else{

        outFC_->Do(operPrm, neighbOpr);

        betweenFC_->Do(operPrm, vector<OperatorBase*> { (OperatorBase*)outFC_ });
    }

    return std::vector<std::string>();
}
 