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
#include "snOperatorCUDA/src/Operator/convolution.h"
#include "snAux/auxFunc.h"
#include "snOperatorCUDA/src/weightInit.h"
#include "snOperatorCUDA/src/activationFunctions.h"
#include "snOperatorCUDA/src/batchNormFunctions.h"
#include "snOperatorCUDA/src/optimizer.h"
#include "snOperatorCUDA/src/structurs.h"
#include "snOperatorCUDA/src/dropOut.h"
#include "snOperatorCUDA/src/cudaCommon.h"

using namespace std;
using namespace SN_Base;

/// convolution layer

Convolution::Convolution(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
    OperatorBase(net, name, node, prms){
        
    load(prms);
}

Convolution::~Convolution(){
        
    setDeviceId(gpuDeviceId_);

    freeParamCUDA(convGPUParams_);   
}

void Convolution::load(std::map<std::string, std::string>& prms){

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
        
    setIntParam("filters", false, true, convPrms_.kernel);
    setIntParam("fWidth", false, false, convPrms_.fWidth);
    setIntParam("fHeight", false, false, convPrms_.fHeight);
    
    if ((prms.find("padding") != prms.end()) && (prms["padding"] == "-1"))
        isPaddingSame_ = true;
    else
        setIntParam("padding", true, false, convPrms_.paddingSet);

    if (prms.find("checkPadding") != prms.end())
        isCheckPadding_ = prms["checkPadding"] == "1";

    if (prms.find("useBias") != prms.end())
        convPrms_.useBias_ = prms["useBias"] == "1";

    setIntParam("stride", false, false, convPrms_.stride);
    setIntParam("dilate", false, false, convPrms_.dilate);
     
    if (prms.find("gpuDeviceId") != prms.end())
        gpuDeviceId_ = stoi(prms["gpuDeviceId"]);

    if (prms.find("batchNorm") != prms.end()){

        string bnType = prms["batchNorm"];
        if (bnType == "none") batchNormType_ = batchNormType::none;
        else if (bnType == "beforeActive") batchNormType_ = batchNormType::beforeActive;
        else if (bnType == "postActive") batchNormType_ = batchNormType::postActive;
        else
            ERROR_MESS("param 'batchNorm' = " + bnType + " indefined");
    }
        
    setInternPrm(prms);
}

bool Convolution::setInternPrm(std::map<std::string, std::string>& prms){

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

bool Convolution::setBatchNorm(const batchNorm& bn){

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

batchNorm Convolution::getBatchNorm()const{

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

std::vector<std::string> Convolution::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
    
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

void Convolution::forward(const SN_Base::Tensor& inTns, const operationParam& operPrm){

    snSize insz = inTns.size();
    inputMem_ = &inTns;

    // Has the size of the data changed?
    if (insz != inSzMem_){
        inSzMem_ = insz;
        updateConfig(operPrm.isLerning, insz);
    }
         
    snFloat* in = inputMem_->getDataGPU(),
           * out = baseOut_.getDataGPU(),
           * weight = baseWeight_.getDataGPU();

    snSize outsz = baseOut_.size();
       
    // calculation
    forwardCUDA(convPrms_, weight, insz, in, outsz, out, convGPUParams_);

    /// dropOut
    if (dropOut_ > 0.F)
        dropOut(operPrm.isLerning, dropOut_, outsz, out);

    // batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
       channelBatchNorm(true, operPrm.isLerning, outsz, out, out, baseBatchNorm_);
        
    /// active function
    if (activeType_ != activeType::none)
       activationForward(outsz, out, activeType_);
           
    // batchNorm
    if (batchNormType_ == batchNormType::postActive)
        channelBatchNorm(true, operPrm.isLerning, outsz, out, out, baseBatchNorm_);
}

void Convolution::backward(const SN_Base::Tensor& inTns, const operationParam& operPrm){
    
    snFloat* gradIn = inTns.getDataGPU(); 
    snSize insz = inTns.size(),
           outsz = baseOut_.size();

    // batchNorm
    if (batchNormType_ == batchNormType::postActive)
        channelBatchNorm(false, true, insz, gradIn, gradIn, baseBatchNorm_);
    
    // active function
    if (activeType_ != activeType::none)        
        activationBackward(outsz, baseOut_.getDataGPU(), gradIn, activeType_);
    
    // batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        channelBatchNorm(false, true, insz, gradIn, gradIn, baseBatchNorm_);
   
    // calculation of the output gradient and weight correction
    snFloat* gradOut = baseGrad_.getDataGPU(),
           * weight = baseWeight_.getDataGPU();
  
    if (!isFreeze_){
        snFloat* dWeight = auxGPUParams_["dWeight"],
               * in = inputMem_->getDataGPU();
       
        // calculation
        backwardCUDA_GW(convPrms_, weight, insz, in, outsz, gradIn, gradOut, dWeight, convGPUParams_);
                       
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
        backwardCUDA_G(convPrms_, weight, insz, outsz, gradIn, gradOut, convGPUParams_);
    }   
}

void Convolution::updateConfig(bool isLern, const snSize& newsz){
    
    size_t& kernel = convPrms_.kernel,
          & fWidth = convPrms_.fWidth,
          & fHeight = convPrms_.fHeight,
          & paddingSet = convPrms_.paddingSet,
          & paddingW = convPrms_.paddingW,
          & paddingH = convPrms_.paddingH,
          & stride = convPrms_.stride,
          & dilate = convPrms_.dilate;

    size_t stp = fWidth * fHeight * newsz.d, ntp = (stp + 1) * kernel;  // + 1 - bias
        
    // leave the existing weights as they are, initialize the remainder
    size_t wcsz = baseWeight_.size().size();
    if (ntp > wcsz){
                
        baseWeight_.resize(snSize(kernel, stp + 1));

        if (wcsz == 0)
           weightInit(baseWeight_, ntp, stp + 1, kernel, weightInitType_);
    }
    
    // +bias?
    /*   if (!useBias_){
    memset(baseWeight_.getDataCPU() + stp * kernel, 0, kernel * sizeof(snFloat));
    }*/

    snSize outSz(0, 0, kernel, newsz.n);
          
    if (isPaddingSame_){
        outSz.w = newsz.w;
        outSz.h = newsz.h;

        paddingW = (newsz.w * (stride - 1) + fWidth + (fWidth - 1) * (dilate - 1) - stride) / 2;
        paddingH = (newsz.h * (stride - 1) + fHeight + (fHeight - 1) * (dilate - 1) - stride) / 2;
    }
    else{
        paddingW = paddingH = paddingSet;

        outSz.w = (newsz.w + paddingW * 2 - fWidth - (fWidth - 1) * (dilate - 1)) / stride + 1;
        outSz.h = (newsz.h + paddingH * 2 - fHeight - (fHeight - 1) * (dilate - 1)) / stride + 1;
    }

    // check correct
    if (isCheckPadding_){
        size_t res = (newsz.w + paddingW * 2 - fWidth - (fWidth - 1) * (dilate - 1)) % stride;
        if (res != 0)
            ERROR_MESS("not correct param 'stride' or 'fWidth'");

        res = (newsz.h + paddingH * 2 - fHeight - (fHeight - 1) * (dilate - 1)) % stride;
        if (res != 0)
            ERROR_MESS("not correct param 'stride' or 'fHeight'");
    }
     
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

    iniParamCUDA(isLern, newsz, outSz, convPrms_, &convGPUParams_);
} 