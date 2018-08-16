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
#include "resize.h"
#include "snAux/auxFunc.h"

using namespace std;
using namespace SN_Base;

/// изменение размера слоя
Resize::Resize(const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(name, node, prms){

    
}

std::vector<std::string> Resize::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
   
    if (basePrms_.find("outSize") == basePrms_.end()){
        ERROR_MESS("not set param 'outSize'");
        return std::vector < std::string > {"noWay"};
    }

    if (neighbOpr.size() > 1){
        ERROR_MESS("neighbOpr.size() > 1");
        return std::vector < std::string > {"noWay"};
    }
    
    Tensor* inTns = nullptr, *outTns = nullptr;
    if (operPrm.action == snAction::forward){
        inTns = neighbOpr[0]->getOutput();
        outTns = baseOut_;
    }
    else{
        inTns = neighbOpr[0]->getGradient();
        outTns = baseGrad_;
    }
          
    snSize nsz, csz = inTns->size();
    auto ss = SN_Aux::split(basePrms_["outSize"], " ");
    if (ss.size() < 4){
        ERROR_MESS("'outSize' args < 4");
        return std::vector < std::string > {"noWay"};
    }
    nsz.w = stol(ss[0]);
    nsz.h = stol(ss[1]);
    nsz.d = stol(ss[2]);
    nsz.n = stol(ss[3]);

    if ((nsz.w != csz.w) || (nsz.h != csz.h) || (nsz.d != csz.d)){
        ERROR_MESS("(nsz.w != csz.w) || (nsz.h != csz.h) || (nsz.d != csz.d)");
        return std::vector < std::string > {"noWay"};
    }

    outTns->setData(inTns->getData(), inTns->size());
    outTns->resize(nsz);

    return std::vector<std::string>();
}
