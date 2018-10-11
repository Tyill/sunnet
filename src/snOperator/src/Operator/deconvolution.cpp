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
#include "snOperator/src/Operator/deconvolution.h"
#include "snAux/auxFunc.h"
#include "snOperator/src/weightInit.h"
#include "snOperator/src/activeFunctions.h"
#include "snOperator/src/optimizer.h"
#include "snOperator/src/structurs.h"
#include "snOperator/src/mathFunctions.h"

using namespace std;
using namespace SN_Base;

/// deconvolution layer
Deconvolution::Deconvolution(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
    OperatorBase(net, name, node, prms){
        
    load(prms);
}

Deconvolution::~Deconvolution(){

    baseInput_ = 0;

    if (calcMode_ == calcMode::CUDA)
        freeParamCUDA(gpuParams_);
    else if (calcMode_ == calcMode::OpenCL)
        freeParamOCL(gpuParams_);
}

void Deconvolution::load(std::map<std::string, std::string>& prms){

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
        
    setIntParam("kernel", false, true, deconvParams_.kernel);
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

    if (prms.find("gpuClearMem") != prms.end())
        gpuClearMem_ = stoi(prms["gpuClearMem"]) == 1;

    if (prms.find("gpuDeviceId") != prms.end())
        gpuDeviceId_ = stoi(prms["gpuDeviceId"]);

    if (prms.find("mode") != prms.end()){

        string mode = prms["mode"];
        if (mode == "CPU") calcMode_ = calcMode::CPU;
        else if (mode == "CUDA") calcMode_ = calcMode::CUDA;
        else if (mode == "OpenCL") calcMode_ = calcMode::OpenCL;
        else
            ERROR_MESS("param 'mode' = " + mode + " indefined");
    }

    // aux arrays
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

bool Deconvolution::setBatchNorm(const batchNorm& bn){

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

std::vector<std::string> Deconvolution::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
       
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
            Tensor tns = *neighbOpr[0]->getGradient();
            for (size_t i = 1; i < neighbOpr.size(); ++i){

                if (tns != *neighbOpr[i]->getGradient()){
                    ERROR_MESS("operators size is not equals");
                    return std::vector < std::string > {"noWay"};
                }
                tns += *neighbOpr[i]->getGradient();
            }
            backward(&tns, operPrm);
        }
    }

    return std::vector<std::string>();
}

void Deconvolution::forward(SN_Base::Tensor* inTns, const operationParam& operPrm){

    snSize insz = inTns->size();
    baseInput_ = inTns;

    if (insz != inSzMem_){
        inSzMem_ = insz;
        updateConfig(insz);
    }

    snFloat* pInTns = inTns->getData();
   
    snFloat* out = baseOut_->getData(), *weight = baseWeight_->getData();
    snSize outsz = baseOut_->size();

    switch (calcMode_){
    case calcMode::CPU:    forwardCPU(deconvParams_, weight, insz, pInTns, outsz, out); break;
    case calcMode::CUDA:   forwardCUDA(deconvParams_, weight, insz, pInTns, outsz, out, gpuParams_); break;
    case calcMode::OpenCL: forwardOCL(deconvParams_, weight, insz, pInTns, outsz, out, gpuParams_); break;
    }

    /// dropOut
    if (dropOut_ > 0.F)
        dropOut(operPrm.isLerning, dropOut_, outsz, out);

    /// batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        calcBatchNorm(true, operPrm.isLerning, outsz, out, out, baseBatchNorm_);
       

    /// active func
    switch (activeType_){
    case activeType::sigmoid:   fv_sigmoid(out, outsz.size()); break;
    case activeType::relu:      fv_relu(out, outsz.size()); break;
    case activeType::leakyRelu: fv_leakyRelu(out, outsz.size()); break;
    case activeType::elu:       fv_elu(out, outsz.size()); break;
    default: break;
    }

    /// batchNorm
    if (batchNormType_ == batchNormType::postActive)
        calcBatchNorm(true, operPrm.isLerning, outsz, out, out, baseBatchNorm_);
    
}

void Deconvolution::backward(SN_Base::Tensor* inTns, const operationParam& operPrm){
    
    snFloat* gradIn = inTns->getData();

    /// batchNorm
    if (batchNormType_ == batchNormType::postActive)
        calcBatchNorm(false, true, inTns->size(), gradIn, gradIn, baseBatchNorm_);
    
    /// active func
    if (activeType_ != activeType::none){

        snFloat* out = baseOut_->getData();
        
        size_t osz = baseOut_->size().size();
        switch (activeType_){
        case activeType::sigmoid:   df_sigmoid(out, osz); break;
        case activeType::relu:      df_relu(out, osz); break;
        case activeType::leakyRelu: df_leakyRelu(out, osz); break;
        case activeType::elu:       df_elu(out, osz); break;
        default: break;
        }

        // update grad
        for (size_t i = 0; i < osz; ++i) gradIn[i] *= out[i];
    }

    /// batchNorm
    if (batchNormType_ == batchNormType::beforeActive)
        calcBatchNorm(false, true, inTns->size(), gradIn, gradIn, baseBatchNorm_);
    
    // calculation of the output gradient and weight correction
    snFloat* grOut = baseGrad_->getData();   
    snFloat* weight = baseWeight_->getData();
    snFloat* in = baseInput_->getData();
    snFloat* out = baseOut_->getData();

    snSize insz = baseInput_->size(), outsz = baseOut_->size();
  
    if (!isFreeze_){
        snFloat* dWeight = auxParams_["dWeight"].data();
        
        switch (calcMode_){
        case calcMode::CPU:    backwardCPU_GW(deconvParams_, weight, insz, in, outsz, gradIn, grOut, dWeight); break;
        case calcMode::CUDA:   backwardCUDA_GW(deconvParams_, weight, insz, in, outsz, gradIn, grOut, dWeight, gpuParams_); break;
        case calcMode::OpenCL: backwardOCL_GW(deconvParams_, weight, insz, in, outsz, gradIn, grOut, dWeight, gpuParams_); break;
        }

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
    else{ // isFreeze
        switch (calcMode_){
        case calcMode::CPU:    backwardCPU_G(deconvParams_, weight, insz, outsz, gradIn, grOut); break;
        case calcMode::CUDA:   backwardCUDA_G(deconvParams_, weight, insz, outsz, gradIn, grOut, gpuParams_); break;
        case calcMode::OpenCL: backwardOCL_G(deconvParams_, weight, insz, outsz, gradIn, grOut, gpuParams_); break;
        }
    }        
}

void Deconvolution::calcBatchNorm(bool fwBw, bool isLern, const snSize& insz, snFloat* in, snFloat* out, batchNorm prm){

    /* Select 1 output layer from each image in the batch and normalize */

    size_t stepD = insz.w * insz.h, stepN = stepD * insz.d, bsz = insz.n;

    if (!isLern){

        for (size_t i = 0; i < insz.d; ++i){

            /// y = ^x * γ + β
            for (size_t j = 0; j < bsz; ++j){

                snFloat* cin = in + stepN * j + stepD * i,
                    *cout = out + stepN * j + stepD * i;
                for (size_t k = 0; k < stepD; ++k)
                    cout[k] = (cin[k] - prm.mean[k]) * prm.scale[k] / prm.varce[k] + prm.schift[k];
            }
            prm.offset(stepD);
        }       
    }
    else{

        snFloat* share = (snFloat*)calloc(stepD * bsz, sizeof(snFloat));
        snSize sz(insz.w, insz.h, 1, insz.n);

        for (size_t i = 0; i < insz.d; ++i){

            snFloat* pSh = share;
            snFloat* pIn = in + stepD * i;
            for (size_t j = 0; j < bsz; ++j){

                memcpy(pSh, pIn, stepD * sizeof(snFloat));
                pSh += stepD;
                pIn += stepN;
            }

            if (fwBw)
                batchNormForward(sz, share, share, prm);
            else
                batchNormBackward(sz, share, share, prm);
              
            pSh = share;
            snFloat* pOut = out + stepD * i;
            for (size_t j = 0; j < bsz; ++j){
                memcpy(pOut, pSh, stepD * sizeof(snFloat));
                pSh += stepD;
                pOut += stepN;
            }

            prm.offset(stepD);
            prm.norm += stepD * bsz;
        }
        free(share);
    }         
}

void Deconvolution::updateConfig(const snSize& newsz){
    
    size_t stp = deconvParams_.fWidth * deconvParams_.fHeight * deconvParams_.kernel,
        ntp = (stp + 1) * newsz.d;
        
    // leave the existing weights as they are, initialize the remainder
    size_t wcsz = baseWeight_->size().size();
    if (ntp > wcsz){
                
        baseWeight_->resize(snSize(newsz.d, stp + 1));
        snFloat* wd = baseWeight_->getData();
        switch (weightInitType_){
        case weightInitType::uniform: wi_uniform(wd + wcsz, ntp - wcsz); break;
        case weightInitType::he: wi_he(wd + wcsz, ntp - wcsz, stp + 1); break;
        case weightInitType::lecun:wi_lecun(wd + wcsz, ntp - wcsz, deconvParams_.kernel); break;
        case weightInitType::xavier:wi_xavier(wd + wcsz, ntp - wcsz, stp + 1, deconvParams_.kernel); break;
        }
    }
        
    snSize outSz(0, 0, deconvParams_.kernel, newsz.n);
        
   
    outSz.w = (newsz.w - 1) * deconvParams_.stride + deconvParams_.fWidth;
    outSz.h = (newsz.h - 1) * deconvParams_.stride + deconvParams_.fHeight;
    
    baseOut_->resize(outSz);
    baseGrad_->resize(newsz);
        
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
        iniParamCUDA(newsz, outSz, deconvParams_, &gpuParams_);
    else if (calcMode_ == calcMode::OpenCL)
        iniParamOCL(newsz, outSz, deconvParams_, &gpuParams_);
} 



