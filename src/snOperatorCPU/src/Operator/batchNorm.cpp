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
#include "snOperatorCPU/src/Operator/batchNorm.h"
#include "snOperatorCPU/src/batchNormFunctions.h"

using namespace std;
using namespace SN_Base;


BatchNorm::BatchNorm(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){
       
    inSzMem_ = snSize(0);
}

bool BatchNorm::setBatchNorm(const batchNorm& bn){

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

std::vector<std::string> BatchNorm::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
      
    if (operPrm.action == snAction::forward){

        if (neighbOpr.size() > 1){
            ERROR_MESS("neighbOpr.size() > 1");
            return std::vector < std::string > {"noWay"};
        }

        baseOut_ = neighbOpr[0]->getOutput();
    
        snFloat* out = baseOut_.getDataCPU();
        snSize outsz = baseOut_.size();

        if (outsz != inSzMem_){
            inSzMem_ = outsz;
            updateConfig(operPrm.isLerning, outsz);
        }
               
        batchNormForward(operPrm.isLerning, outsz, out, out, baseBatchNorm_);
                
    }
    else{ // backward
       
        baseGrad_ = neighbOpr[0]->getGradient();

        for (size_t i = 1; i < neighbOpr.size(); ++i){

            if (baseGrad_ != neighbOpr[i]->getGradient()){
                ERROR_MESS("operators size is not equals");
                return std::vector < std::string > {"noWay"};
            }
            baseGrad_ += neighbOpr[i]->getGradient();
        }
               
        snFloat* out = baseGrad_.getDataCPU();
      
        batchNormBackward(baseGrad_.size(), out, out, baseBatchNorm_);
    }
    
    return vector<string>();
}

void BatchNorm::updateConfig(bool isLern, const snSize& newsz){
        
    size_t osz = newsz.w * newsz.h * newsz.d;

    auxParams_["bn_mean"] = vector<snFloat>(osz, 0);         baseBatchNorm_.mean = auxParams_["bn_mean"].data();
    auxParams_["bn_varce"] = vector<snFloat>(osz, 1);        baseBatchNorm_.varce = auxParams_["bn_varce"].data();
    auxParams_["bn_scale"] = vector<snFloat>(osz, 1);        baseBatchNorm_.scale = auxParams_["bn_scale"].data();
    auxParams_["bn_schift"] = vector<snFloat>(osz, 0);       baseBatchNorm_.schift = auxParams_["bn_schift"].data();
 
    if (isLern){
        auxParams_["bn_norm"] = vector<snFloat>(osz * newsz.n);  baseBatchNorm_.norm = auxParams_["bn_norm"].data();
        auxParams_["bn_dScale"] = vector<snFloat>(osz, 0);       baseBatchNorm_.dScale = auxParams_["bn_dScale"].data();
        auxParams_["bn_dSchift"] = vector<snFloat>(osz, 0);      baseBatchNorm_.dSchift = auxParams_["bn_dSchift"].data();
    }
    baseBatchNorm_.sz = newsz;
    baseBatchNorm_.sz.n = 1;
}