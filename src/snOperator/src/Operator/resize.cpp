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
Resize::Resize(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){

    baseOut_ = new Tensor();
    baseGrad_ = new Tensor();
}

std::vector<std::string> Resize::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
       
    if (neighbOpr.size() > 1){
        ERROR_MESS("neighbOpr.size() > 1");
        return std::vector < std::string > {"noWay"};
    }
      
    if (operPrm.action == snAction::forward){

        if (basePrms_.find("outDiapByN") == basePrms_.end()){
            ERROR_MESS("not set param 'outDiapByN'");
            return std::vector < std::string > {"noWay"};
        }
                      
        auto ss = SN_Aux::split(basePrms_["outDiapByN"], " ");
        if (ss.size() < 2){
            ERROR_MESS("'outDiapByN' args < 2");
            return std::vector < std::string > {"noWay"};
        }
        bgDiapN_ = max(0, stoi(ss[0]));
        endDiapN_ = max(0, stoi(ss[1]));
        
        if (bgDiapN_ > endDiapN_){
            ERROR_MESS("'outDiapByN' bgDiapN > endDiapN");
            return std::vector < std::string > {"noWay"};
        }

        Tensor* inTns = neighbOpr[0]->getOutput(), *outTns = baseOut_;

        snSize csz = inTns->size();
        inSizeMem_ = csz;
        
        if ((csz.n < bgDiapN_) || (csz.n < endDiapN_)){
            ERROR_MESS("'outDiapByN' (csz.n < bgDiapN_) || (csz.n < endDiapN_)");
            return std::vector < std::string > {"noWay"};
        }

        csz.n = endDiapN_ - bgDiapN_;
       
        outTns->setData(inTns->getData() + bgDiapN_ * csz.w * csz.h * csz.d, csz);
    }
    else{
        Tensor* inTns = neighbOpr[0]->getGradient(), *outTns = baseGrad_;
                      
        outTns->resize(inSizeMem_);
      
        snSize csz = inTns->size();
        size_t stp = csz.w * csz.h * csz.d;
        outTns->setData(inTns->getData() + bgDiapN_ * stp, csz);

        // остаток обнуляем
        memset(outTns->getData(), 0, bgDiapN_ * stp * sizeof(snFloat));
        memset(outTns->getData() + endDiapN_ * stp, 0, (inSizeMem_.n - endDiapN_) * stp * sizeof(snFloat));
    }

    return std::vector<std::string>();
}
