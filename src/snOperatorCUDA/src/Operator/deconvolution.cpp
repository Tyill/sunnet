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
#include "snOperatorCUDA/src/Operator/deconvolution.h"
#include "snAux/auxFunc.h"
#include "snOperatorCUDA/src/weightInit.h"
#include "snOperatorCUDA/src/activationFunctions.h"
#include "snOperatorCUDA/src/optimizer.h"
#include "snOperatorCUDA/src/structurs.h"
#include "snOperatorCUDA/src/cudaCommon.h"
#include "snOperatorCUDA/src/dropOut.h"
#include "snOperatorCUDA/src/batchNormFunctions.h"

using namespace std;
using namespace SN_Base;

/// deconvolution layer
Deconvolution::Deconvolution(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
    OperatorBase(net, name, node, prms){
        
    load(prms);
}

Deconvolution::~Deconvolution(){
       
    setDeviceId(gpuDeviceId_);

    freeParamCUDA(gpuParams_);   
}

void Deconvolution::load(std::map<std::string, std::string>& prms){
    
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
        
    setIntParam("filters", false, true, deconvParams_.kernel);
    setIntParam("fWidth", false, false, deconvParams_.fWidth);
    setIntParam("fHeight", false, false, deconvParams_.fHeight);
  
    setIntParam("stride", false, false, deconvParams_.stride);
  
    if (prms.find("batchNorm") != prms.end()){

        string bnType = prms["batchNorm"];
        if (bnType == "none") batchNormType_ = batchNormType::none;
        else if (bnType == "beforeActive") batchNormType_ = batchNormType::beforeActive;
        else if (bnType == "postActive") batchNormType_ = batchNormType::postActive;
        else
            ERROR_MESS("param 'batchNorm' = " + bnType + " indefined");
    }
       
    if (prms.find("gpuDeviceId") != prms.end())
        gpuDeviceId_ = stoi(prms["gpuDeviceId"]);
            
   
    setInternPrm(prms);
}

bool Deconvolution::setInternPrm(std::map<std::string, std::string>& prms){

    basePrms_ = prms;

    if (prms.find("active") != prms.end()){

        string atype = prms["active"];
        if (atype == "none") activeType_ = activeType::none;
        else if (atype == "sigmoid") activeType_ = activeType::sigmoid;
        else if (atype == "relu") activeType_ = activeType::relu;
        else if (atype == "leakyRelu") activeType_ = activeType::leakyRelu;
        else if (atype == "elu") activeType_ = activeType::elu;
        else
            ERROR_MESS("param 'active' = " + atype + " indefined");
    }

    if (prms.find("optimizer") != prms.end()){

        string optType = prms["optimizer"];
        if (optType == "sgd") optimizerType_ = optimizerType::sgd;
        else if (optType == "sgdMoment") optimizerType_ = optimizerType::sgdMoment;
        else if (optType == "adagrad") optimizerType_ = optimizerType::adagrad;
        else if (optType == "adam") optimizerType_ = optimizerType::adam;
        else if (optType == "RMSprop") optimizerType_ = optimizerType::RMSprop;
        else
            ERROR_MESS("param 'optimizer' = " + optType + " indefined");
    }

    if (prms.find("weightInit") != prms.end()){

        string wInit = prms["weightInit"];
        if (wInit == "uniform") weightInitType_ = weightInitType::uniform;
        else if (wInit == "he") weightInitType_ = weightInitType::he;
        else if (wInit == "lecun") weightInitType_ = weightInitType::lecun;
        else if (wInit == "xavier") weightInitType_ = weightInitType::xavier;
        else
            ERROR_MESS("param 'weightInit' = " + wInit + " indefined");
    }
       
    if (prms.find("decayMomentDW") != prms.end())
        optDecayMomentDW_ = stof(prms["decayMomentDW"]);

    if (prms.find("decayMomentWGr") != prms.end())
        optDecayMomentWGr_ = stof(prms["decayMomentWGr"]);

    if (prms.find("lmbRegular") != prms.end())
        optLmbRegular_ = stof(prms["lmbRegular"]);

    if (prms.find("batchNormLr") != prms.end())
        baseBatchNorm_.lr = stof(prms["batchNormLr"]);

    if (prms.find("freeze") != prms.end())
        isFreeze_ = prms["freeze"] == "1";
                
    if (prms.find("dropOut") != prms.end()){
        dropOut_ = stof(prms["dropOut"]);
        if (dropOut_ > 1.F) dropOut_ = 1.F;
        else if (dropOut_ < 0.F) dropOut_ = 0.F;
    }

    return true;
}

bool Deconvolution::setBatchNorm(const batchNorm& bn){

    size_t csz = baseBatchNorm_.sz.size(),
        osz = bn.sz.size();

    baseBatchNorm_.mean = memRealloc(csz, osz, baseBatchNorm_.mean, 0);
    baseBatchNorm_.varce = memRealloc(csz, osz, baseBatchNorm_.varce, 1);
    baseBatchNorm_.scale = memRealloc(csz, osz, baseBatchNorm_.scale, 1);
    baseBatchNorm_.schift = memRealloc(csz, osz, baseBatchNorm_.schift, 0);

    memCpyCPU2GPU(osz, baseBatchNorm_.mean, osz, bn.mean);
    memCpyCPU2GPU(osz, baseBatchNorm_.varce, osz, bn.varce);
    memCpyCPU2GPU(osz, baseBatchNorm_.scale, osz, bn.scale);
    memCpyCPU2GPU(osz, baseBatchNorm_.schift, osz, bn.schift);

    baseBatchNorm_.sz = bn.sz;

    return true;
}

SN_Base::batchNorm Deconvolution::getBatchNorm()const{

    size_t csz = baseBatchNorm_.sz.size();

    auxCPUParams_["bn_mean"] = vector<snFloat>(csz);
    auxCPUParams_["bn_varce"] = vector<snFloat>(csz);
    auxCPUParams_["bn_scale"] = vector<snFloat>(csz);
    auxCPUParams_["bn_schift"] = vector<snFloat>(csz);

    memCpyGPU2CPU(csz, auxCPUParams_["bn_mean"].data(), csz, baseBatchNorm_.mean);
    memCpyGPU2CPU(csz, auxCPUParams_["bn_varce"].data(), csz, baseBatchNorm_.varce);
    memCpyGPU2CPU(csz, auxCPUParams_["bn_scale"].data(), csz, baseBatchNorm_.scale);
    memCpyGPU2CPU(csz, auxCPUParams_["bn_schift"].data(), csz, baseBatchNorm_.schift);

    batchNorm bn;
    bn.mean = auxCPUParams_["bn_mean"].data();
    bn.varce = auxCPUParams_["bn_varce"].data();
    bn.scale = auxCPUParams_["bn_scale"].data();
    bn.schift = auxCPUParams_["bn_schift"].data();

    bn.sz = csz;

    return bn;
}

std::vector<std::string> Deconvolution::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
       
    setDeviceId(gpuDeviceId_);

    if (operPrm.action == snAction::forward){

        if (neighbOpr.size() > 1){
            ERROR_MESS("neighbOpr.size() > 1");
            return std::vector < std::string > {"noWay"};
        }

        forward(neighbOpr[0]->getOutput(), operPrm);
    }
    else{
        if (neighbOpr.size() == 1){
            backward(neighbOpr[0]->getGradient(), operPrm);
        }
        else{
            Tensor tns = neighbOpr[0]->getGradient();
            for (size_t i = 1; i < neighbOpr.size(); ++i){

                if (tns != neighbOpr[i]->getGradient()){
                    ERROR_MESS("operators size is not equals");
                    return std::vector < std::string > {"noWay"};
                }
                tns += neighbOpr[i]->getGradient();
            }
            backward(tns, operPrm);
        }
    }

    return std::vector<std::string>();
}

void Deconvolution::forward(const SN_Base::Tensor& inTns, const operationParam& operPrm){

    snSize insz = inTns.size();
    inputMem_ = &inTns;

    if (insz != inSzMem_){
        inSzMem_ = insz;
        updateConfig(operPrm.isLerning, insz);
    }

    snFloat* pInTns = inTns.getDataGPU();
   
    snFloat* out = baseOut_.getDataGPU(), *weight = baseWeight_.getDataGPU();
    snSize outsz = baseOut_.size();

    // calculation
    forwardCUDA(deconvParams_, weight, insz, pInTns, outsz, out, gpuParams_);
   
    /// dropOut
    if (dropOut_ > 0.F)
        dropOut(operPrm.isLerning, dropOut_, outsz, out);

    /// batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        channelBatchNorm(true, operPrm.isLerning, outsz, out, out, baseBatchNorm_);
       
    // active func
    activationForward(outsz, out, activeType_);
    
    // batchNorm
    if (batchNormType_ == batchNormType::postActive)
        channelBatchNorm(true, operPrm.isLerning, outsz, out, out, baseBatchNorm_);
    
}

void Deconvolution::backward(const SN_Base::Tensor& inTns, const operationParam& operPrm){
    
    snFloat* gradIn = inTns.getDataGPU();

    // batchNorm
    if (batchNormType_ == batchNormType::postActive)
        channelBatchNorm(false, true, inTns.size(), gradIn, gradIn, baseBatchNorm_);
    
    /// active func
    if (activeType_ != activeType::none){
                         
        activationBackward(baseOut_.size(), baseOut_.getDataGPU(), gradIn, activeType_);        
    }

    // batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        channelBatchNorm(false, true, inTns.size(), gradIn, gradIn, baseBatchNorm_);
    
    // calculation of the output gradient and weight correction
    snFloat* grOut = baseGrad_.getDataGPU();   
    snFloat* weight = baseWeight_.getDataGPU();
    snFloat* in = inputMem_->getDataGPU();
    snFloat* out = baseOut_.getDataGPU();

    snSize insz = inputMem_->size(), outsz = baseOut_.size();
  
    if (!isFreeze_){
        snFloat* dWeight = auxGPUParams_["dWeight"];
        
        // calculation
        backwardCUDA_GW(deconvParams_, weight, insz, in, outsz, gradIn, grOut, dWeight, gpuParams_);        

        snFloat* dWPrev = auxGPUParams_["dWPrev"];
        snFloat* dWGrad = auxGPUParams_["dWGrad"];
        size_t wsz = baseWeight_.size().size();
               
        optimizer(dWeight,
            dWPrev,
            dWGrad,
            weight,
            wsz,
            operPrm.lr,
            optLmbRegular_,
            optDecayMomentDW_,
            optDecayMomentWGr_,
            optimizerType_);
    }
    else{ // isFreeze
        backwardCUDA_G(deconvParams_, weight, insz, outsz, gradIn, grOut, gpuParams_);
       
    }        
}

void Deconvolution::updateConfig(bool isLern, const snSize& newsz){
    
    size_t stp = deconvParams_.fWidth * deconvParams_.fHeight * deconvParams_.kernel,
        ntp = (stp + 1) * newsz.d;
        
    // leave the existing weights as they are, initialize the remainder
    size_t wcsz = baseWeight_.size().size();
    if (ntp > wcsz){
                
        baseWeight_.resize(snSize(newsz.d, stp + 1));
       
        if (wcsz == 0)
          weightInit(baseWeight_, ntp - wcsz, stp + 1, deconvParams_.kernel, weightInitType_);
    }
    
    // +bias?
    /*   if (!useBias_){
    memset(baseWeight_.getDataCPU() + stp * kernel, 0, kernel * sizeof(snFloat));
    }*/

    snSize outSz(0, 0, deconvParams_.kernel, newsz.n);
        
   
    outSz.w = (newsz.w - 1) * deconvParams_.stride + deconvParams_.fWidth;
    outSz.h = (newsz.h - 1) * deconvParams_.stride + deconvParams_.fHeight;
    
    baseOut_.resize(outSz);

    if (isLern){
        baseGrad_.resize(newsz);

        // aux array
        auxGPUParams_["dWeight"] = memRealloc(0, ntp, auxGPUParams_["dWeight"], 0);
        auxGPUParams_["dWPrev"] = memRealloc(0, ntp, auxGPUParams_["dWPrev"], 0);
        auxGPUParams_["dWGrad"] = memRealloc(0, ntp, auxGPUParams_["dWGrad"], 0);
    }

    size_t csz = baseBatchNorm_.sz.w * baseBatchNorm_.sz.h * baseBatchNorm_.sz.d,
           osz = outSz.w * outSz.h * outSz.d;
    
    if (batchNormType_ != batchNormType::none){        
        baseBatchNorm_.mean = memRealloc(csz, osz, baseBatchNorm_.mean, 0);
        baseBatchNorm_.varce = memRealloc(csz, osz, baseBatchNorm_.varce, 1);
        baseBatchNorm_.scale = memRealloc(csz, osz, baseBatchNorm_.scale, 1);
        baseBatchNorm_.schift = memRealloc(csz, osz, baseBatchNorm_.schift, 0);

        if (isLern){
            baseBatchNorm_.norm = memRealloc(csz * outSz.n, osz * outSz.n, baseBatchNorm_.mean, 0);
            baseBatchNorm_.dScale = memRealloc(csz, osz, baseBatchNorm_.dScale, 0);
            baseBatchNorm_.dSchift = memRealloc(csz, osz, baseBatchNorm_.dSchift, 0);
        }

        baseBatchNorm_.sz = outSz;
        baseBatchNorm_.sz.n = 1;
    }  

    iniParamCUDA(isLern, newsz, outSz, deconvParams_, &gpuParams_);
    
} 



