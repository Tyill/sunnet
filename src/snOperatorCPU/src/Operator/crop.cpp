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
#include "snOperatorCPU/src/Operator/crop.h"
#include "snAux/auxFunc.h"

using namespace std;
using namespace SN_Base;


/// Trimming data
Crop::Crop(void* net, const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(net, name, node, prms){
       
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

        baseOut_ = neighbOpr[0]->getOutput();
                   
        auto nsz = SN_Aux::split(basePrms_["roi"], " ");

        if (nsz.size() != 4){
            ERROR_MESS("'roi' param no correct. Must be four arguments: x y w h");
            return vector < string > {"noWay"};
        }

        baseSz_ = baseOut_.size();
      
        size_t x = max<size_t>(0, min<size_t>(stoi(nsz[0]), baseSz_.w - 1)),
               y = max<size_t>(0, min<size_t>(stoi(nsz[1]), baseSz_.h - 1)),
               w = max<size_t>(0, min<size_t>(stoi(nsz[2]), baseSz_.w - x)),
               h = max<size_t>(0, min<size_t>(stoi(nsz[3]), baseSz_.h - y));
        
        roi_ = roi(x, y, w, h);

        Tensor tns(snSize(w, h, baseSz_.d, baseSz_.n));

        snFloat* src = baseOut_.getDataCPU(),
               * dst = tns.getDataCPU();

        copyTo(true, roi_, baseSz_, src, dst);

        baseOut_ = tns;
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
                
        Tensor tns(baseSz_);

        snFloat* dst = tns.getDataCPU(),
               * src = baseGrad_.getDataCPU();
                
        copyTo(false, roi_, baseSz_, dst, src);

        baseGrad_ = tns;
    }
    
    return vector<string>();
}

void Crop::copyTo(bool inToOut, const roi& roi, const snSize& srcSz, snFloat* in, snFloat* out){
       
    size_t bsz = srcSz.d * srcSz.n, srcStp = srcSz.w * srcSz.h, dstStp = roi.w * roi.h;

    if (inToOut){
        for (size_t i = 0; i < bsz; ++i){
            
            snFloat* pIn = in + roi.x + roi.y * srcSz.w + srcStp * i;
            snFloat* pOut = out + dstStp * i;

            for (size_t j = 0; j < roi.h; ++j){
                            
                for (size_t k = 0; k < roi.w; ++k)
                    pOut[k] = pIn[k];  

                pIn += srcSz.w;
                pOut += roi.w;
            }
        }
    }
    else{
        for (size_t i = 0; i < bsz; ++i){

            snFloat* pIn = in + roi.x + roi.y * srcSz.w + srcStp * i;
            snFloat* pOut = out + dstStp * i;

            for (size_t j = 0; j < roi.h; ++j){
                                
                for (size_t k = 0; k < roi.w; ++k)
                    pIn[k] = pOut[k];    

                pIn += srcSz.w;
                pOut += roi.w;
            }            
        }
    }
}

