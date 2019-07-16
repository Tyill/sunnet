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
#pragma once

#include "snBase/snBase.h"
#include"snOperatorCPU/src/structurs.h"
      
/// fullyConnected layer
class FullyConnected final : SN_Base::OperatorBase{

public:

    FullyConnected(void* net, const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

    ~FullyConnected();

    std::vector<std::string> Do(const SN_Base::operationParam&, const std::vector<OperatorBase*>& neighbOpr) override;
        
    bool setInternPrm(std::map<std::string, std::string>& prms) override;
        
    bool setBatchNorm(const SN_Base::batchNorm& bn) override;

private:
        
    size_t kernel_ = 10;                                      ///< number of hidden neurons
                                                              
    activeType activeType_ = activeType::relu;                ///< active func type
    optimizerType optimizerType_ = optimizerType::adam;       ///< optimizer weight type
    weightInitType weightInitType_ = weightInitType::he;      ///< init weight
    batchNormType batchNormType_ = batchNormType::none;       ///< batchNorm 
    SN_Base::snSize inSzMem_;                                 ///< input sz mem
     
    const SN_Base::Tensor* inputMem_ = nullptr;

    bool useBias_ = true;
    bool isFreeze_ = false;                                   ///< not change weight
   
    SN_Base::snFloat dropOut_ = 0.F;                          ///< rand off output
  
    SN_Base::snFloat optDecayMomentDW_ = 0.9F,               ///< optimizer weight coef
                     optDecayMomentWGr_ = 0.99F,
                     optLmbRegular_ = 0.001F;

    
    std::map<std::string, std::vector<SN_Base::snFloat>> auxParams_;  ///< aux data
        

    void load(std::map<std::string, std::string>& prms);

    void updateConfig(bool isLern, const SN_Base::snSize& newSz);
      
    void forward(const SN_Base::Tensor& inTns, const SN_Base::operationParam& operPrm);
    void backward(const SN_Base::Tensor& inTns, const SN_Base::operationParam& operPrm);

    /// CPU ///////////////////////////
    
    void forwardCPU(size_t kernel,   
        const SN_Base::snSize& insz, 
        const SN_Base::snFloat* input,
        const SN_Base::snFloat* weight,
        SN_Base::snFloat* output);   
      
    // calc grad and dw
    void backwardCPU_GW(size_t kernel, 
        const SN_Base::snFloat* weight,
        const SN_Base::snSize& insz,   
        const SN_Base::snFloat* input,
        const SN_Base::snFloat* gradIn,
        SN_Base::snFloat* gradOut,     
        SN_Base::snFloat* dWeightOut); 
             
    // calc grad
    void backwardCPU_G(size_t kernel,
        const SN_Base::snFloat* weight,
        const SN_Base::snSize& insz, 
        const SN_Base::snFloat* gradIn,
        SN_Base::snFloat* gradOut);  
    
};