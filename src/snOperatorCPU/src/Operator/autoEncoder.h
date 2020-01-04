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
#include "snOperatorCPU/src/Operator/fullyConnected.h"
      
/// AE layer
class AutoEncoder final : SN_Base::OperatorBase{

public:

    AutoEncoder(void* net, const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

    ~AutoEncoder();

    std::vector<std::string> Do(const SN_Base::operationParam&, const std::vector<OperatorBase*>& neighbOpr) override;
        
    bool setInternPrm(std::map<std::string, std::string>& prms) override;
        
    bool setBatchNorm(const SN_Base::batchNorm& bn) override;
      
    bool setWeight(const SN_Base::snFloat* data, const SN_Base::snSize& dsz) override;
        
    batchNorm getBatchNorm() const override;

    const SN_Base::Tensor& getWeight() const override;

    const SN_Base::Tensor& getOutput() const override;

    const SN_Base::Tensor& getGradient() const override;

private:
    
    SN_Base::snSize inSzMem_;                                 ///< input sz mem

    mutable Tensor weight_;
       
    mutable std::map<std::string, std::vector<SN_Base::snFloat>> auxParams_;  ///< aux data

    FullyConnected* betweenFC_ = nullptr,
                  * outFC_ = nullptr;    
};