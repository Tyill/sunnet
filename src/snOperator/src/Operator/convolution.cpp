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
#include "snOperator/src/Operator/convolution.h"
#include "snAux/auxFunc.h"
#include "snOperator/src/weightInit.h"
#include "snOperator/src/activeFunctions.h"
#include "snOperator/src/optimizer.h"
#include "snOperator/src/structurs.h"
#include "snOperator/src/mathFunctions.h"

using namespace std;
using namespace SN_Base;

/// convolution layer

Convolution::Convolution(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
    OperatorBase(net, name, node, prms){
        
    load(prms);
}

Convolution::~Convolution(){
        
    if (calcMode_ == calcMode::CUDA)
        freeParamCUDA(gpuParams_);
    else if (calcMode_ == calcMode::OpenCL)
        freeParamOCL(gpuParams_);
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
        
    setIntParam("kernel", false, true, convPrms_.kernel);
    setIntParam("fWidth", false, false, convPrms_.fWidth);
    setIntParam("fHeight", false, false, convPrms_.fHeight);
    
    if ((prms.find("padding") != prms.end()) && (prms["padding"] == "-1"))
        isPaddingSame_ = true;
    else
        setIntParam("padding", true, false, convPrms_.paddingSet);

    if (prms.find("checkPadding") != prms.end())
        isCheckPadding_ = prms["checkPadding"] == "1";

    setIntParam("stride", false, false, convPrms_.stride);
    setIntParam("dilate", false, false, convPrms_.dilate);

    if (prms.find("gpuClearMem") != prms.end())
        gpuClearMem_ = stoi(prms["gpuClearMem"]) == 1;
   
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

    if (prms.find("mode") != prms.end()){

        string mode = prms["mode"];
        if (mode == "CPU") calcMode_ = calcMode::CPU;
        else if (mode == "CUDA") calcMode_ = calcMode::CUDA;
        else if (mode == "OpenCL") calcMode_ = calcMode::OpenCL;
        else
            ERROR_MESS("param 'mode' = " + mode + " indefined");
    }

    // aux array
    auxParams_["dWeight"] = vector<snFloat>();
    auxParams_["dWPrev"] = vector<snFloat>();
    auxParams_["dWGrad"] = vector<snFloat>();    

    if (batchNormType_ != batchNormType::none){
        auxParams_["bn_mean"] = vector<snFloat>();
        auxParams_["bn_varce"] = vector<snFloat>();
        auxParams_["bn_scale"] = vector<snFloat>();
        auxParams_["bn_schift"] = vector<snFloat>();
        auxParams_["bn_norm"] = vector<snFloat>();
        auxParams_["bn_dScale"] = vector<snFloat>();
        auxParams_["bn_dSchift"] = vector<snFloat>();
        auxParams_["bn_onc"] = vector<snFloat>();
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
        opt_decayMomentDW_ = stof(prms["decayMomentDW"]);

    if (prms.find("decayMomentWGr") != prms.end())
        opt_decayMomentWGr_ = stof(prms["decayMomentWGr"]);

    if (prms.find("lmbRegular") != prms.end())
        opt_lmbRegular_ = stof(prms["lmbRegular"]);

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

    size_t osz = bn.sz.size();

    auxParams_["bn_mean"] = vector<snFloat>(osz, 0);     baseBatchNorm_.mean = auxParams_["bn_mean"].data();
    auxParams_["bn_varce"] = vector<snFloat>(osz, 1);    baseBatchNorm_.varce = auxParams_["bn_varce"].data();
    auxParams_["bn_scale"] = vector<snFloat>(osz, 1);    baseBatchNorm_.scale = auxParams_["bn_scale"].data();
    auxParams_["bn_schift"] = vector<snFloat>(osz, 0);   baseBatchNorm_.schift = auxParams_["bn_schift"].data();

    memcpy(baseBatchNorm_.mean, bn.mean, osz * sizeof(snFloat));
    memcpy(baseBatchNorm_.varce, bn.varce, osz * sizeof(snFloat));
    memcpy(baseBatchNorm_.scale, bn.scale, osz * sizeof(snFloat));
    memcpy(baseBatchNorm_.schift, bn.schift, osz * sizeof(snFloat));

    baseBatchNorm_.sz = bn.sz;

    return true;
}

std::vector<std::string> Convolution::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
       
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

    /// Has the size of the data changed?
    if (insz != inSzMem_){
        inSzMem_ = insz;
        updateConfig(insz, inDataExpSz_);
    }

    /// copy with offset padding for each image
    snFloat* in = inputMem_->getData();
    bool isSame = (convPrms_.paddingW == 0) && (convPrms_.paddingH == 0);
    if (!isSame){
        inTnsExp_.resize(inDataExpSz_);       
        paddingOffs(false, convPrms_.paddingW, convPrms_.paddingH, insz, inTnsExp_.getData(), in);
        in = inTnsExp_.getData();
    }
    
    /// calculation of the output values
    snFloat* out = baseOut_.getData(), *weight = baseWeight_.getData();
    snSize outsz = baseOut_.size();

    switch (calcMode_){
    case calcMode::CPU:    forwardCPU(convPrms_, weight, inDataExpSz_, in, outsz, out); break;
    case calcMode::CUDA:   forwardCUDA(convPrms_, weight, inDataExpSz_, in, outsz, out, gpuParams_); break;
    case calcMode::OpenCL: forwardOCL(convPrms_, weight, inDataExpSz_, in, outsz, out, gpuParams_); break;
    }

    if (!operPrm.isLerning && !isSame)
        inTnsExp_.tfree();
       
    /// dropOut
    if (dropOut_ > 0.F)
        dropOut(operPrm.isLerning, dropOut_, outsz, out);

    /// batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        channelBatchNorm(true, operPrm.isLerning, outsz, out, out, baseBatchNorm_);
        
    /// active function
    activeFuncForward(outsz.size(), out, activeType_);
           
    /// batchNorm
    if (batchNormType_ == batchNormType::postActive)
        channelBatchNorm(true, operPrm.isLerning, outsz, out, out, baseBatchNorm_);
}

void Convolution::backward(const SN_Base::Tensor& inTns, const operationParam& operPrm){
    
    snFloat* gradIn = inTns.getData();
 
    /// batchNorm
    if (batchNormType_ == batchNormType::postActive)
        channelBatchNorm(false, true, inTns.size(), gradIn, gradIn, baseBatchNorm_);
    
    // active function
    if (activeType_ != activeType::none){

        snFloat* out = baseOut_.getData();
        
        size_t osz = baseOut_.size().size();
        activeFuncBackward(osz, out, activeType_);

        // update grad
        for (size_t i = 0; i < osz; ++i) gradIn[i] *= out[i];
    }

    /// batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        channelBatchNorm(false, true, inTns.size(), gradIn, gradIn, baseBatchNorm_);
    
    // calculation of the output gradient and weight correction
    snFloat* gradOut = baseGrad_.getData();
      
    bool isSame = (convPrms_.paddingW == 0) && (convPrms_.paddingH == 0);
    if (!isSame){
        gradOutExp_.resize(inDataExpSz_);
        gradOut = gradOutExp_.getData();
    }

    snFloat* weight = baseWeight_.getData();
  
    if (!isFreeze_){
        snFloat* dWeight = auxParams_["dWeight"].data();
        
        snFloat* in = inputMem_->getData();
        if (!isSame)
            in = inTnsExp_.getData();
        
        switch (calcMode_){
        case calcMode::CPU:    backwardCPU_GW(convPrms_, weight, inDataExpSz_, in, baseOut_.size(), gradIn, gradOut, dWeight); break;
        case calcMode::CUDA:   backwardCUDA_GW(convPrms_, weight, inDataExpSz_, in, baseOut_.size(), gradIn, gradOut, dWeight, gpuParams_); break;
        case calcMode::OpenCL: backwardOCL_GW(convPrms_, weight, inDataExpSz_, in, baseOut_.size(), gradIn, gradOut, dWeight, gpuParams_); break;
        }
               
        // correct weight
        snFloat* dWPrev = auxParams_["dWPrev"].data();
        snFloat* dWGrad = auxParams_["dWGrad"].data();
        size_t wsz = baseWeight_.size().size();
              
        optimizer(dWeight,
            dWPrev,
            dWGrad,
            weight,
            wsz,
            operPrm.lr,
            opt_lmbRegular_,
            opt_decayMomentDW_,
            opt_decayMomentWGr_,
            optimizerType_);
    }
    else{ // isFreeze
        switch (calcMode_){
        case calcMode::CPU:    backwardCPU_G(convPrms_, weight, inDataExpSz_, baseOut_.size(), gradIn, gradOut); break;
        case calcMode::CUDA:   backwardCUDA_G(convPrms_, weight, inDataExpSz_, baseOut_.size(), gradIn, gradOut, gpuParams_); break;
        case calcMode::OpenCL: backwardOCL_G(convPrms_, weight, inDataExpSz_, baseOut_.size(), gradIn, gradOut, gpuParams_); break;
        }
    }

    if (!isSame){
        paddingOffs(true, convPrms_.paddingW, convPrms_.paddingH, inSzMem_, gradOut, baseGrad_.getData());
        gradOutExp_.tfree();
        inTnsExp_.tfree();
    }
}

void Convolution::updateConfig(const snSize& newsz, SN_Base::snSize& expSz){
    
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
        snFloat* wd = baseWeight_.getData();
        weightInit(wd + wcsz, ntp - wcsz, stp + 1, kernel, weightInitType_);     
    }
        
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

    expSz = snSize(newsz.w + paddingW * 2, newsz.h + paddingH * 2, newsz.d, newsz.n);
      
    baseOut_.resize(outSz);
    baseGrad_.resize(newsz);
        
    // aux array
    auxParams_["dWeight"].resize(ntp, 0);
    auxParams_["dWPrev"].resize(ntp, 0);
    auxParams_["dWGrad"].resize(ntp, 0);

    size_t osz = outSz.w * outSz.h * outSz.d;
    
    if (batchNormType_ != batchNormType::none){        
        auxParams_["bn_mean"].resize(osz, 0);         baseBatchNorm_.mean = auxParams_["bn_mean"].data();
        auxParams_["bn_varce"].resize(osz, 1);        baseBatchNorm_.varce = auxParams_["bn_varce"].data();
        auxParams_["bn_scale"].resize(osz, 1);        baseBatchNorm_.scale = auxParams_["bn_scale"].data();
        auxParams_["bn_schift"].resize(osz, 0);       baseBatchNorm_.schift = auxParams_["bn_schift"].data();
        auxParams_["bn_norm"].resize(osz * outSz.n);  baseBatchNorm_.norm = auxParams_["bn_norm"].data();
        auxParams_["bn_dScale"].resize(osz, 0);       baseBatchNorm_.dScale = auxParams_["bn_dScale"].data();
        auxParams_["bn_dSchift"].resize(osz, 0);      baseBatchNorm_.dSchift = auxParams_["bn_dSchift"].data();
        auxParams_["bn_onc"].resize(outSz.n, 1.F);    baseBatchNorm_.onc = auxParams_["bn_onc"].data();
        baseBatchNorm_.sz = outSz;
        baseBatchNorm_.sz.n = 1;
    }  

    if (calcMode_ == calcMode::CUDA)
        iniParamCUDA(expSz, outSz, convPrms_, &gpuParams_);
    else if (calcMode_ == calcMode::OpenCL)
        iniParamOCL(expSz, outSz, convPrms_, &gpuParams_);
} 