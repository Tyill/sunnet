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
#include "snOperatorCUDA/src/Operator/concat.h"
#include "snAux/auxFunc.h"

using namespace std;
using namespace SN_Base;


/// Compound layers
Concat::Concat(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){
           
    if (basePrms_.find("sequence") == basePrms_.end())
        ERROR_MESS("non set param 'sequence'");    
}

std::vector<std::string> Concat::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
      
    if (operPrm.action == snAction::forward){
                       
        auto seq = SN_Aux::split(basePrms_["sequence"], " ");

        OperatorBase* neighb = nullptr;
        for (size_t j = 0; j < neighbOpr.size(); ++j){
            if (neighbOpr[j]->node() == seq[0]){
                neighb = neighbOpr[j];
                break;
            }
        }

        if (!neighb){
            ERROR_MESS("not found neighbor '" + seq[0] + "'");
            return vector<string>{"noWay"};
        }

        baseOut_ = neighb->getOutput();

        size_t ssz = seq.size();
        for (size_t i = 1; i < ssz; ++i){

            OperatorBase* neighb = nullptr;
            for (size_t j = 0; j < neighbOpr.size(); ++j){
                if (neighbOpr[j]->node() == seq[i]){
                    neighb = neighbOpr[j];
                    break;
                }
            }

            if (!neighb){
                ERROR_MESS("not found neighbor '" + seq[i] + "'");
                return vector<string>{"noWay"};
            }

            snSize csz = baseOut_.size();

            snSize nbsz = neighb->getOutput().size();

            if ((csz.w != nbsz.w) || (csz.h != nbsz.h) || (csz.n != nbsz.n)){
                ERROR_MESS("operators size is not equals");
                return std::vector < std::string > {"noWay"};
            }

            Tensor buff = baseOut_;

            baseOut_.resize(snSize(csz.w, csz.h, csz.d + nbsz.d, csz.n));

            size_t sz = csz.w * csz.h * (csz.d + nbsz.d),
                 cstp = csz.w * csz.h * csz.d,
                 nstp = nbsz.w * nbsz.h * nbsz.d;
            for (size_t j = 0; j < csz.n; ++j){

                snFloat* dst = baseOut_.getDataGPU() + sz * j,
                    *src = buff.getDataGPU() + cstp * j;

                memcpy(dst, src, cstp * sizeof(snFloat));


                dst += cstp;
                src = neighb->getOutput().getDataGPU() + nstp * j;

                memcpy(dst, src, nstp * sizeof(snFloat));
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
    }
    
    return vector<string>();
}
