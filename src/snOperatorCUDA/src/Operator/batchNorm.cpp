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
#include "../cudaCommon.h"
#include "../batchNormFunctions.h"
#include "snOperatorCUDA/src/Operator/batchNorm.h"
#include "snAux/auxFunc.h"

using namespace std;
using namespace SN_Base;


BatchNorm::BatchNorm(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){

   
    inSzMem_ = snSize(0);
}

bool BatchNorm::setBatchNorm(const batchNorm& bn){

    const snSize& csz = baseBatchNorm_.sz,
                  osz = bn.sz;

    baseBatchNorm_.mean = cuMemRealloc(csz, osz, baseBatchNorm_.mean, 0.F);
    baseBatchNorm_.varce = cuMemRealloc(csz, osz, baseBatchNorm_.varce, 1.F);
    baseBatchNorm_.scale = cuMemRealloc(csz, osz, baseBatchNorm_.scale, 1.F);
    baseBatchNorm_.schift = cuMemRealloc(csz, osz, baseBatchNorm_.schift, 0.F);

    cuMemCpyCPU2GPU(osz, baseBatchNorm_.mean, bn.mean);
    cuMemCpyCPU2GPU(osz, baseBatchNorm_.varce, bn.varce);
    cuMemCpyCPU2GPU(osz, baseBatchNorm_.scale, bn.scale);
    cuMemCpyCPU2GPU(osz, baseBatchNorm_.schift, bn.schift);

    baseBatchNorm_.sz = bn.sz;

    return true;
}

batchNorm BatchNorm::getBatchNorm()const{

    const snSize& csz = baseBatchNorm_.sz;

    auxCPUParams_["bn_mean"] = vector<snFloat>(csz.size());
    auxCPUParams_["bn_varce"] = vector<snFloat>(csz.size());
    auxCPUParams_["bn_scale"] = vector<snFloat>(csz.size());
    auxCPUParams_["bn_schift"] = vector<snFloat>(csz.size());

    cuMemCpyGPU2CPU(csz, auxCPUParams_["bn_mean"].data(), baseBatchNorm_.mean);
    cuMemCpyGPU2CPU(csz, auxCPUParams_["bn_varce"].data(), baseBatchNorm_.varce);
    cuMemCpyGPU2CPU(csz, auxCPUParams_["bn_scale"].data(), baseBatchNorm_.scale);
    cuMemCpyGPU2CPU(csz, auxCPUParams_["bn_schift"].data(), baseBatchNorm_.schift);

    batchNorm bn;
    bn.mean = auxCPUParams_["bn_mean"].data();
    bn.varce = auxCPUParams_["bn_varce"].data();
    bn.scale = auxCPUParams_["bn_scale"].data();
    bn.schift = auxCPUParams_["bn_schift"].data();

    bn.sz = csz;

    return bn;
}

std::vector<std::string> BatchNorm::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
      
    if (operPrm.action == snAction::forward){

        if (neighbOpr.size() > 1){
            ERROR_MESS("neighbOpr.size() > 1");
            return std::vector < std::string > {"noWay"};
        }

        baseOut_ = neighbOpr[0]->getOutput();
    
        snFloat* out = baseOut_.getDataGPU();
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
               
        snFloat* out = baseGrad_.getDataGPU();
        
        batchNormBackward(baseGrad_.size(), out, out, baseBatchNorm_);
    }
    
    return vector<string>();
}

void BatchNorm::updateConfig(bool isLern, const snSize& newsz){
        
    const snSize& csz = baseBatchNorm_.sz,
                  osz = snSize(newsz.w, newsz.h, newsz.d);

    baseBatchNorm_.mean = cuMemRealloc(csz, osz, baseBatchNorm_.mean, 0.F);
    baseBatchNorm_.varce = cuMemRealloc(csz, osz, baseBatchNorm_.varce, 1.F);
    baseBatchNorm_.scale = cuMemRealloc(csz, osz, baseBatchNorm_.scale, 1.F);
    baseBatchNorm_.schift = cuMemRealloc(csz, osz, baseBatchNorm_.schift, 0.F);

    if (isLern){
        baseBatchNorm_.norm = cuMemRealloc(snSize(0), snSize(newsz.w, newsz.h, newsz.d, newsz.n), baseBatchNorm_.norm, 0.F);
        baseBatchNorm_.dScale = cuMemRealloc(snSize(0), osz, baseBatchNorm_.dScale, 0.F);
        baseBatchNorm_.dSchift = cuMemRealloc(snSize(0), osz, baseBatchNorm_.dSchift, 0.F);
    }

    baseBatchNorm_.sz = osz;
}