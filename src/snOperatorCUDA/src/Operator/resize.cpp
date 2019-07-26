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
#include "snOperatorCUDA/src/Operator/resize.h"
#include "snAux/auxFunc.h"

using namespace std;
using namespace SN_Base;


/// Trimming the number of layers
Resize::Resize(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){
       
}

std::vector<std::string> Resize::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
      
    if (operPrm.action == snAction::forward){

        baseOut_ = neighbOpr[0]->getOutput();
                
        if (neighbOpr.size() > 1){
            ERROR_MESS("neighbOpr.size() > 1");
            return std::vector < std::string > {"noWay"};
        }
        
        if (basePrms_.find("fwdDiap") != basePrms_.end()){

            snSize csz = baseOut_.size();

            auto ls = SN_Aux::split(basePrms_["fwdDiap"], " ");

            size_t begin = min<size_t>(stol(ls[0]), csz.d - 1), end = csz.d;
            
            if (ls.size() > 1) end = min<size_t>(stol(ls[1]), csz.d);
            
            if (begin >= end){
                ERROR_MESS("fwdDiap begin >= end");
                return std::vector < std::string > {"noWay"};
            }

            if ((end - begin) < csz.d){

                Tensor buff = baseOut_;

                baseOut_.resize(snSize(csz.w, csz.h, (end - begin), csz.n));

                size_t sz = csz.w * csz.h * (end - begin),
                       offset = csz.w * csz.h * begin,
                       cstp = csz.w * csz.h * csz.d;
                for (size_t j = 0; j < csz.n; ++j){

                    snFloat* dst = baseOut_.getDataGPU() + sz * j,
                           * src = buff.getDataGPU() + cstp * j + offset;

                    cuMemCpyGPU2GPU(sz, dst, src, true);
                }
            }
        }
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

        if (basePrms_.find("bwdDiap") != basePrms_.end()){

            snSize csz = baseGrad_.size();

            auto ls = SN_Aux::split(basePrms_["bwdDiap"], " ");

            size_t begin = min<size_t>(stol(ls[0]), csz.d - 1), end = csz.d;

            if (ls.size() > 1) end = min<size_t>(stol(ls[1]), csz.d);

            if (begin >= end){
                ERROR_MESS("bwdDiap begin >= end");
                return std::vector < std::string > {"noWay"};
            }

            if ((end - begin) < csz.d){

                Tensor buff = baseGrad_;

                baseGrad_.resize(snSize(csz.w, csz.h, (end - begin), csz.n));

                size_t sz = csz.w * csz.h * (end - begin),
                    offset = csz.w * csz.h * begin,
                    cstp = csz.w * csz.h * csz.d;
                for (size_t j = 0; j < csz.n; ++j){

                    snFloat* dst = baseGrad_.getDataGPU() + sz * j,
                           * src = buff.getDataGPU() + cstp * j + offset;

                    cuMemCpyGPU2GPU(sz, dst, src, true);
                }
            }
        }
    }
    
    return vector<string>();
}
