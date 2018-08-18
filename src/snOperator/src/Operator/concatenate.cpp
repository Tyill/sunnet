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
#include "concatenate.h"

using namespace std;
using namespace SN_Base;

/// объединение 2х слоев
Concatenate::Concatenate(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){

    
}

std::vector<std::string> Concatenate::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
    
    if (neighbOpr.size() == 2){

        Tensor* tnsOne = nullptr, *tnsTwo = nullptr, *tnsOut = nullptr;
        if (operPrm.action == snAction::forward){
            tnsOne = neighbOpr[0]->getOutput();
            tnsTwo = neighbOpr[1]->getOutput();
            tnsOut = baseOut_;
        }
        else{
            tnsOne = neighbOpr[0]->getGradient();
            tnsTwo = neighbOpr[1]->getGradient();
            tnsOut = baseGrad_;
        }

        snSize fsz = tnsOne->size();
        snSize ssz = tnsTwo->size();
            
        if ((fsz.w != ssz.w) || (fsz.h != ssz.h) || (fsz.d != ssz.d)){
            ERROR_MESS("(fsz.w != ssz.w) || (fsz.h != ssz.h) || (fsz.d != ssz.d)")
            return std::vector<std::string>{"noWay"};
        }

        tnsOut->resize(snSize(fsz.w, fsz.h, fsz.d, fsz.n + ssz.n));
            
        memcpy(tnsOut->getData(),
            tnsOne->getData(),
            tnsOne->size().size());

        memcpy(tnsOut->getData() + tnsOne->size().size(),
                tnsTwo->getData(),
                tnsTwo->size().size());
        
    }
    else if (neighbOpr.size() == 1){
        if (operPrm.action == snAction::forward){
            auto nb = neighbOpr[0]->getOutput();
            baseOut_->setData(nb->getData(), nb->size());
        }
        else{
            auto nb = neighbOpr[0]->getGradient();
            baseGrad_->setData(nb->getData(), nb->size());
        }
    }
    else{
        ERROR_MESS("neighbOpr > 2");

        return std::vector<std::string>{"noWay"};
    }
    
    return std::vector<std::string>();
}
