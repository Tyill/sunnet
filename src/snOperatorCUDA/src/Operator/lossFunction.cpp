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
#include "../lossFunctions.h"
#include "snOperatorCUDA/src/Operator/lossFunction.h"

using namespace std;
using namespace SN_Base;


void LossFunction::load(std::map<std::string, std::string>& prms){
    
    if (prms.find("loss") != prms.end()){

        string stype = prms["loss"];
        if (stype == "softMaxToCrossEntropy")
            lossType_ = lossType::softMaxACrossEntropy;
        else if (stype == "binaryCrossEntropy")
            lossType_ = lossType::binaryCrossEntropy;
        else if (stype == "regressionMSE")
            lossType_ = lossType::regressionMSE;
        else if (stype == "userLoss")
            lossType_ = lossType::userLoss;
        else
            ERROR_MESS("param 'loss' = " + stype + " indefined");
    }
    else
        ERROR_MESS("not found param 'loss'");
}

LossFunction::LossFunction(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){

    load(prms);
}


std::vector<std::string> LossFunction::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){

    if (neighbOpr.size() > 1){
        ERROR_MESS("neighbOpr.size() > 1");
        return std::vector < std::string > {"noWay"};
    }

    if (lossType_ == lossType::userLoss){
        if (basePrms_.find("cbackName") == basePrms_.end()){
            ERROR_MESS("not set param 'cbackName'");
            return std::vector < std::string > {"noWay"};
        }
    }

    if (operPrm.action == snAction::forward)
        forward(neighbOpr[0]->getOutput());
    else
        backward(neighbOpr[0]->getGradient(), operPrm);

    return std::vector<std::string>();
}

void LossFunction::forward(const Tensor& inTns){

    baseOut_ = inTns;    
   
    if (lossType_ != lossType::userLoss){

        lossForward(baseOut_.size(), baseOut_.getDataGPU(), lossType_);
    }
    else{
        snSize outSz;
        snFloat* outData = nullptr;
        g_userCBack(this, basePrms_["cbackName"], node_,
            true, inTns.size(), inTns.getDataCPU(), outSz, &outData);

        if (outData){
            baseOut_.setDataCPU(outData, outSz);
        }
        else
            ERROR_MESS("not set 'outData' in userCBack");
    }    
}

void LossFunction::backward(const Tensor& inTns, const operationParam& operPrm){

    snSize tsz = inTns.size();        
   
    snSize grsz = baseGrad_.size();
    if (grsz != tsz)
        baseGrad_.resize(tsz);
        
    auto smOut = baseOut_.getDataGPU();    
    auto target = inTns.getDataGPU();      
    auto grad = baseGrad_.getDataGPU();  

    switch (lossType_){
    case lossType::softMaxACrossEntropy:{

        size_t nsz = tsz.size();
        for (size_t i = 0; i < nsz; ++i)
            grad[i] = smOut[i] - target[i];

        break;
    }

    case lossType::binaryCrossEntropy:{

        size_t nsz = tsz.size();
        for (size_t i = 0; i < nsz; ++i)
            grad[i] = (smOut[i] - target[i]) / (smOut[i] * (1.F - smOut[i]));
    
        break;
    }
    // Mean Square Error
    case lossType::regressionMSE:{
                
        size_t nsz = tsz.size();
        for (size_t i = 0; i < nsz; ++i){
                        
            grad[i] = 2 * (smOut[i] - target[i]) / nsz;
        }

        break;
    }

    case lossType::userLoss:{

        snSize outSz;
        snFloat* outData = nullptr;
        g_userCBack(this, basePrms_["cbackName"], node_,
            false, inTns.size(), inTns.getDataCPU(), outSz, &outData);

        if (outData){
            baseGrad_.setDataCPU(outData, outSz);
        }
        else
            ERROR_MESS("not set 'outData' in userCBack");

        break;
    }


    default:
        ERROR_MESS("param 'lossType' indefined");
        break;
    }
}
