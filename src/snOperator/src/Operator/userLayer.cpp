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
#include "userLayer.h"

using namespace std;
using namespace SN_Base;

/// польз слой
UserLayer::UserLayer(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){

    baseOut_ = new Tensor();
    baseGrad_ = new Tensor();
}

std::vector<std::string> UserLayer::Do(const operationParam& opr, const std::vector<OperatorBase*>& neighbOpr){
        
    if (neighbOpr.size() > 1){
        ERROR_MESS("neighbOpr.size() > 1");
        return std::vector < std::string > {"noWay"};
    }

    if (basePrms_.find("cbackName") == basePrms_.end()){
        ERROR_MESS("not set param 'cbackName'");
        return std::vector < std::string > {"noWay"};
    }

    Tensor* inTns = nullptr,* outTns = nullptr;
    bool isAct = true;
    if (opr.action == SN_Base::snAction::forward){
        inTns = neighbOpr[0]->getOutput();
        outTns = baseOut_;
    }
    else{
        inTns = neighbOpr[0]->getGradient();
        outTns = baseGrad_;
        isAct = false;
    }

    snSize outSz;
    snFloat* outData = nullptr;

    g_userCBack(this, basePrms_["cbackName"], node_,
        isAct, inTns->size(), inTns->getData(), outSz, &outData);
    
    if (outData)
       outTns->setData(outData, outSz);
    else{
        ERROR_MESS("not set 'outData' in userCBack");
        return std::vector < std::string > {"noWay"};
    }

    return std::vector<std::string>();
}
