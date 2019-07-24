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
#include "snOperatorCUDA/src/Operator/fullyConnected.h"
#include "snAux/auxFunc.h"
#include "snOperatorCUDA/src/weightInit.h"
#include "snOperatorCUDA/src/activationFunctions.h"
#include "snOperatorCUDA/src/optimizer.h"
#include "snOperatorCUDA/src/structurs.h"
#include "snOperatorCUDA/src/batchNormFunctions.h"
#include "snOperatorCUDA/src/cudaCommon.h"
#include "snOperatorCUDA/src/dropOut.h"

using namespace std;
using namespace SN_Base;

/// fullyConnected layer
FullyConnected::FullyConnected(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
    OperatorBase(net, name, node, prms){
        
    load(prms);
}

FullyConnected::~FullyConnected(){
    
    setDeviceId(gpuDeviceId_);

    freeParamCUDA(gpuParams_);    
}

void FullyConnected::load(std::map<std::string, std::string>& prms){
        
    if ((prms.find("units") != prms.end()) && SN_Aux::is_number(prms["units"])){

        size_t kernel = stoi(prms["units"]);
        if (kernel > 0)
            kernel_ = kernel;
        else
            ERROR_MESS("param 'units' <= 0");
    }
    else
        ERROR_MESS("not found (or not numder) param 'units'");

    if (prms.find("useBias") != prms.end())
        useBias_ = prms["useBias"] == "1";

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

    baseOut_.resize(snSize(kernel_));
  
    // currrect params
    setInternPrm(prms);  

    if (batchNormType_ != batchNormType::none){

        baseBatchNorm_.mean = memRealloc(0, kernel_, baseBatchNorm_.mean, 0);
        baseBatchNorm_.varce = memRealloc(0, kernel_, baseBatchNorm_.varce, 1);
        baseBatchNorm_.scale = memRealloc(0, kernel_, baseBatchNorm_.scale, 1);
        baseBatchNorm_.schift = memRealloc(0, kernel_, baseBatchNorm_.schift, 0);
             
        baseBatchNorm_.sz = snSize(kernel_);
    }
}

bool FullyConnected::setInternPrm(std::map<std::string, std::string>& prms){

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

bool FullyConnected::setBatchNorm(const batchNorm& bn){

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

batchNorm FullyConnected::getBatchNorm()const{

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

std::vector<std::string> FullyConnected::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
    
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

void FullyConnected::forward(const SN_Base::Tensor& inTns, const operationParam& operPrm){

    snSize insz = inTns.size();
    inputMem_ = &inTns;

    if (insz != inSzMem_){
        inSzMem_ = insz;
        updateConfig(operPrm.isLerning, insz);
    }
    
    /// calculation of the output values of neurons
    snFloat* out = baseOut_.getDataGPU(), 
           * weight = baseWeight_.getDataGPU();
  
    // calculation
    forwardCUDA(kernel_, insz, inTns.getDataGPU(), weight, out, gpuParams_);
   
    /// dropOut
    snSize outsz = baseOut_.size();
    if (dropOut_ > 0.F)
        dropOut(operPrm.isLerning, dropOut_, outsz, out);
    
    /// batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        layerBatchNorm(true, operPrm.isLerning, outsz, out, out, baseBatchNorm_);
    
    /// active func
    activationForward(outsz, out, activeType_);
       
    /// batchNorm
    if (batchNormType_ == batchNormType::postActive)
        layerBatchNorm(true, operPrm.isLerning, outsz, out, out, baseBatchNorm_);
}

void FullyConnected::backward(const SN_Base::Tensor& inTns, const operationParam& operPrm){

    snFloat* gradIn = inTns.getDataGPU();

    /// batchNorm
    snSize gsz = inTns.size();
    if (batchNormType_ == batchNormType::postActive)
        layerBatchNorm(false, true, gsz, gradIn, gradIn, baseBatchNorm_);
      
    // active func
    if (activeType_ != activeType::none)                
        activationBackward(baseOut_.size(), baseOut_.getDataGPU(), gradIn, activeType_); 

    // batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        layerBatchNorm(false, true, gsz, gradIn, gradIn, baseBatchNorm_);
      
    // calculation of the output gradient and weight correction
    snFloat* gradOut = baseGrad_.getDataGPU();
    snFloat* weight = baseWeight_.getDataGPU();
       
    if (!isFreeze_){
                
        snFloat* dWeight = auxGPUParams_["dWeight"];
       
        // calculation
        backwardCUDA_GW(kernel_, weight, inSzMem_, inputMem_->getDataGPU(), gradIn, gradOut, dWeight, gpuParams_);
                        
        // correct weight              
        optimizer(dWeight,
                  auxGPUParams_["dWPrev"],
                  auxGPUParams_["dWGrad"],
                  weight,
                  baseWeight_.size(),
                  operPrm.lr,
                  optLmbRegular_,
                  optDecayMomentDW_,
                  optDecayMomentWGr_,
                  optimizerType_);
      
    }
    else{ // isFreeze
        backwardCUDA_G(kernel_, weight, inSzMem_, gradIn, gradOut, gpuParams_);       
    }   
}

void FullyConnected::updateConfig(bool isLern, const snSize& newsz){
    
    size_t stp = newsz.w * newsz.h * newsz.d, ntp = (stp + 1) * kernel_;
        
    // leave the existing weights as they are, initialize the remainder
    size_t wcsz = baseWeight_.size().size();
    if (ntp > wcsz){
                
        baseWeight_.resize(snSize(kernel_, stp + 1));
        
        if (wcsz == 0)
            weightInit(baseWeight_, ntp - wcsz, stp + 1, kernel_, weightInitType_);
    }
    
    // +bias?
 /*   if (!useBias_){
        memset(baseWeight_.getDataCPU() + stp * kernel, 0, kernel * sizeof(snFloat));
    }*/

    baseOut_.resize(snSize(kernel_, 1, 1, newsz.n));

    if (isLern){
        baseGrad_.resize(newsz);

        // aux array
        auxGPUParams_["dWeight"] = memRealloc(0, ntp, auxGPUParams_["dWeight"], 0);
        auxGPUParams_["dWPrev"] = memRealloc(0, ntp, auxGPUParams_["dWPrev"], 0);
        auxGPUParams_["dWGrad"] = memRealloc(0, ntp, auxGPUParams_["dWGrad"], 0);

        if (batchNormType_ != batchNormType::none){

            auto bcsz = baseBatchNorm_.sz.size();

            baseBatchNorm_.norm = memRealloc(bcsz, newsz.n * kernel_, baseBatchNorm_.norm, 0);
            baseBatchNorm_.dScale = memRealloc(bcsz, kernel_, baseBatchNorm_.dScale, 0);
            baseBatchNorm_.dSchift = memRealloc(bcsz, kernel_, baseBatchNorm_.dSchift, 0);
            
            baseBatchNorm_.sz = snSize(kernel_);
        }
    }

    iniParamCUDA(isLern, newsz, kernel_, &gpuParams_);   
} 