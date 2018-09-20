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
#include "crop.h"
#include "snAux/auxFunc.h"

using namespace std;
using namespace SN_Base;


/// Обрезка данных
Crop::Crop(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){

    baseOut_ = new Tensor();
    baseGrad_ = new Tensor();

    if (basePrms_.find("roi") != basePrms_.end()){

        auto nsz = SN_Aux::split(basePrms_["roi"], " ");

        if (nsz.size() != 4)
            ERROR_MESS("'roi' param no correct. Must be four arguments: x y w h");
    }
    else
        ERROR_MESS("no set param 'roi'");
}

std::vector<std::string> Crop::Do(const operationParam& operPrm, const std::vector<OperatorBase*>& neighbOpr){
      
    if (operPrm.action == snAction::forward){

        if (neighbOpr.size() > 1){
            ERROR_MESS("neighbOpr.size() > 1");
            return std::vector < std::string > {"noWay"};
        }

        *baseOut_ = *neighbOpr[0]->getOutput();
                   
        auto nsz = SN_Aux::split(basePrms_["roi"], " ");

        if (nsz.size() != 4){
            ERROR_MESS("'roi' param no correct. Must be four arguments: x y w h");
            vector < string > {"noWay"};
        }

        baseSz_ = baseOut_->size();
      
        size_t x = max<size_t>(0, min<size_t>(stoi(nsz[0]), baseSz_.w - 1)),
               y = max<size_t>(0, min<size_t>(stoi(nsz[1]), baseSz_.h - 1)),
               w = max<size_t>(0, min<size_t>(stoi(nsz[2]), baseSz_.w - x)),
               h = max<size_t>(0, min<size_t>(stoi(nsz[3]), baseSz_.h - y));
        
        roi_ = roi(x, y, w, h);

        Tensor tmpTns(snSize(w, h, baseSz_.d, baseSz_.n));

        snFloat* src = baseOut_->getData(),
               * dst = tmpTns.getData();

        size_t sz = baseSz_.d * baseSz_.n, bstp = baseSz_.w * baseSz_.h, nstp = w * h;       
        for (size_t i = 0; i < sz; ++i)
            copyTo(true, baseSz_.w, baseSz_.h, roi_, src + bstp * i, dst + nstp * i);

        *baseOut_ = tmpTns; 
    }
    else{ // backward
       
        *baseGrad_ = *neighbOpr[0]->getGradient();

        for (size_t i = 1; i < neighbOpr.size(); ++i){

            if (*baseGrad_ != *neighbOpr[i]->getGradient()){
                ERROR_MESS("operators size is not equals");
                return std::vector < std::string > {"noWay"};
            }
            *baseGrad_ += *neighbOpr[i]->getGradient();
        }
                
        Tensor tmpTns(baseSz_);

        snFloat* dst = tmpTns.getData(),
               * src = baseGrad_->getData();

        snSize csz = baseGrad_->size();

        size_t sz = baseSz_.d * baseSz_.n, bstp = baseSz_.w * baseSz_.h, nstp = csz.w * csz.h;
        for (size_t i = 0; i < sz; ++i)
            copyTo(false, baseSz_.w, baseSz_.h, roi_, dst + bstp * i, src + nstp * i);

        *baseGrad_ = tmpTns;
    }
    
    return vector<string>();
}

void Crop::copyTo(bool inToOut, size_t w, size_t h, const roi& roi, snFloat* in, snFloat* out){

    in += roi.x + roi.y * w;

    if (inToOut)
        for (size_t i = 0; i < roi.h; ++i){

            for (size_t j = 0; j < roi.w; ++j)
                out[j] = in[j];

            in += w * i;
            out += roi.w * i;
        }
    else
        for (size_t i = 0; i < roi.h; ++i){

            for (size_t j = 0; j < roi.w; ++j)
                in[j] = out[j];

            in += w * i;
            out += roi.w * i;
        }
}

