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
#include "snOperator/src/Operator/pooling.h"
#include "snAux/auxFunc.h"
#include "snOperator/src/structurs.h"

using namespace std;
using namespace SN_Base;

/// pooling layer
Pooling::Pooling(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
    OperatorBase(net, name, node, prms){
        
    load(prms);
}

Pooling::~Pooling(){

    if (calcMode_ == calcMode::CUDA)
        freeParamCUDA(gpuParams_);
    else  if (calcMode_ == calcMode::OpenCL)
        freeParamOCL(gpuParams_);
}

void Pooling::load(std::map<std::string, std::string>& prms){
   
       
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
    
    setIntParam("kernel", false, false, poolPrms_.kernel);
    
    if (prms.find("stride") != prms.end())
        setIntParam("stride", false, false, poolPrms_.stride);
    else
        poolPrms_.stride = poolPrms_.kernel;

    if (prms.find("gpuClearMem") != prms.end())
        gpuClearMem_ = stoi(prms["gpuClearMem"]) == 1;

    if (prms.find("gpuDeviceId") != prms.end())
        gpuDeviceId_ = stoi(prms["gpuDeviceId"]);

    if (prms.find("pool") != prms.end()){

        string atype = prms["pool"];
        if (atype == "max") poolPrms_.type = poolType::max;
        else if (atype == "avg") poolPrms_.type = poolType::avg;
        else
            ERROR_MESS("param 'pool' = " + atype + " indefined");
    }

    if (prms.find("mode") != prms.end()){

        string mode = prms["mode"];
        if (mode == "CPU") calcMode_ = calcMode::CPU;
        else if (mode == "CUDA") calcMode_ = calcMode::CUDA;
        else if (mode == "OpenCL") calcMode_ = calcMode::OpenCL;
        else
            ERROR_MESS("param 'mode' = " + mode + " indefined");
    }

    basePrms_ = prms;
}

std::vector<std::string> Pooling::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
    
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

void Pooling::forward(const SN_Base::Tensor& inTns, const SN_Base::operationParam& operPrm){

    snSize insz = inTns.size();
    inputMem_ = &inTns;

    if (insz != inSzMem_){
        inSzMem_ = insz;
        updateConfig(operPrm.isLerning, insz);
    }
       
    /// copy with offset padding for each image
    snFloat* in = inputMem_->getData();
    if (isPadding_){
        inTnsExp_.resize(inDataExpSz_);
        paddingOffs(false, poolPrms_.paddingW, poolPrms_.paddingH, insz, inTnsExp_.getData(), inputMem_->getData());
        insz = inDataExpSz_;
        in = inTnsExp_.getData();
    }

    /// output calculation
    snFloat* out = baseOut_.getData();
   
    switch (calcMode_){
    case calcMode::CPU:    forwardCPU(poolPrms_, insz, in, baseOut_.size(), out, outInx_.data()); break;
    case calcMode::CUDA:   forwardCUDA(poolPrms_, insz, in, baseOut_.size(), out, outInx_.data(), gpuParams_); break;
    case calcMode::OpenCL: forwardOCL(poolPrms_, insz, in, baseOut_.size(), out, outInx_.data(), gpuParams_); break;
    }      

    if (!operPrm.isLerning && isPadding_)
        inTnsExp_.tfree();
}

void Pooling::backward(const SN_Base::Tensor& inTns, const operationParam& operPrm){

    snFloat* gradIn = inTns.getData();
        
    snFloat* input = inputMem_->getData();
    snFloat* gradOut = baseGrad_.getData();
    if (isPadding_){
        gradOutExp_.resize(inDataExpSz_);
        gradOut = gradOutExp_.getData();
        input = inTnsExp_.getData();
    }

    /// grad calculation
    switch (calcMode_){
    case calcMode::CPU:    backwardCPU(poolPrms_, baseOut_.size(), outInx_.data(), gradIn, inDataExpSz_, gradOut); break;
    case calcMode::CUDA:   backwardCUDA(poolPrms_, baseOut_.size(), outInx_.data(), baseOut_.getData(), gradIn, inDataExpSz_, input, gradOut, gpuParams_); break;
    case calcMode::OpenCL: backwardOCL(poolPrms_, baseOut_.size(), outInx_.data(), gradIn, inDataExpSz_, gradOut, gpuParams_); break;
    }
   
    if (isPadding_){
        paddingOffs(true, poolPrms_.paddingW, poolPrms_.paddingH, inSzMem_, gradOut, baseGrad_.getData());
        gradOutExp_.tfree();
    }
}

void Pooling::updateConfig(bool isLern, const snSize& newsz){
    
    size_t& kernel = poolPrms_.kernel,
          & paddingW = poolPrms_.paddingW,
          & paddingH = poolPrms_.paddingH,
          & stride = poolPrms_.stride;

    snSize outSz(0, 0, newsz.d, newsz.n);
   
    outSz.w = (newsz.w - kernel) / stride + 1;
    outSz.h = (newsz.h - kernel) / stride + 1;

    // check correct
    size_t resW = (newsz.w - kernel) % stride,
           resH = (newsz.h - kernel) % stride;
    
    isPadding_ = (resW != 0) || (resH != 0);

    inDataExpSz_ = newsz;

    if (isPadding_){   
       
        paddingW = 1;
        paddingH = 1;

        outSz.w = (newsz.w + paddingW * 2 - kernel) / stride + 1;
        outSz.h = (newsz.h + paddingH * 2 - kernel) / stride + 1;

        inDataExpSz_ = snSize(newsz.w + paddingW * 2, newsz.h + paddingH * 2, newsz.d, newsz.n);
    }
        
    baseOut_.resize(outSz);

    outInx_.resize(outSz.size(), 0);
    
    if (isLern)
       baseGrad_.resize(newsz);
    
    if (calcMode_ == calcMode::CUDA)
        iniParamCUDA(isLern, inDataExpSz_, outSz, poolPrms_, &gpuParams_);
    else if (calcMode_ == calcMode::OpenCL)
        iniParamOCL(isLern, inDataExpSz_, outSz, poolPrms_, &gpuParams_);
}