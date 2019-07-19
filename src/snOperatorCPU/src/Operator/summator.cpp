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
#include "snOperatorCPU/src/Operator/summator.h"

using namespace std;
using namespace SN_Base;

/// adder of 2 and more layers
Summator::Summator(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){
    
    if (basePrms_.find("type") != basePrms_.end()){
        if (basePrms_["type"] == "summ") sType_ = sType::summ;
        else if (basePrms_["type"] == "diff") sType_ = sType::diff;
        else if (basePrms_["type"] == "mean") sType_ = sType::mean;
        else
            ERROR_MESS("param 'type' indefined");
    }
}

std::vector<std::string> Summator::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
        
        
    if (operPrm.action == snAction::forward){

        baseOut_ = neighbOpr[0]->getOutput();

        size_t sz = neighbOpr.size();
        for (size_t i = 1; i < sz; ++i){
            
            if (baseOut_ != neighbOpr[i]->getOutput()){
                ERROR_MESS("operators size is not equals");
                return std::vector < std::string > {"noWay"};
            }  
            switch (sType_){
            case Summator::sType::summ: baseOut_ += neighbOpr[i]->getOutput(); break;
            case Summator::sType::diff: baseOut_ -= neighbOpr[i]->getOutput(); break;
            case Summator::sType::mean: mean(baseOut_, neighbOpr[i]->getOutput()); break;
            }                
        }
    }
    else{

        baseGrad_ = neighbOpr[0]->getGradient();

        size_t sz = neighbOpr.size();
        for (size_t i = 1; i < sz; ++i){
             
            if (baseGrad_ != neighbOpr[i]->getGradient()){
                ERROR_MESS("operators size is not equals");
                return std::vector < std::string > {"noWay"};
            }
            baseGrad_ += neighbOpr[i]->getGradient();                
        }
    }
    

    return std::vector<std::string>();
}

void Summator::mean(SN_Base::Tensor& inout, const SN_Base::Tensor& two){
   
    snFloat* done = inout.getDataCPU(),
           * dtwo = two.getDataCPU();

    size_t sz = inout.size().size();
    for (size_t i = 0; i < sz; ++i){
        done[i] = (done[i] + dtwo[i]) / 2;
    }
}