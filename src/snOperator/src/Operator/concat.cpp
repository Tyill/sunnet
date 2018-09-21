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
#include "concat.h"
#include "snAux/auxFunc.h"

using namespace std;
using namespace SN_Base;


/// Соединение слоев
Concat::Concat(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){

    baseOut_ = new Tensor();
    baseGrad_ = new Tensor();
       
}

std::vector<std::string> Concat::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
      
    if (operPrm.action == snAction::forward){

       Tensor out = *neighbOpr[0]->getOutput();

       snSize csz = out.size();
       for (size_t i = 1; i < neighbOpr.size(); ++i){

           snSize nbsz = neighbOpr[i]->getOutput()->size();

           if ((csz.w != nbsz.w) || (csz.h != nbsz.h) || (csz.n != nbsz.n)){
               ERROR_MESS("operators size is not equals");
               return std::vector < std::string > {"noWay"};
           }

           out.resize(snSize(csz.w, csz.h, csz.d + nbsz.d, csz.n));

        /*   for (size_t j = 0; j < csz.n; ++j){
               memcpy(out.getData() + csz.w * csz.h * csz.d * j,
                      neighbOpr[i]->getOutput()->getData() + nbsz.w * nbsz.h * nbsz.d * j,
                        );
           }
           csz.d += nbsz.d;*/
       }

    }
    else{ // backward

    }
    
    return vector<string>();
}
