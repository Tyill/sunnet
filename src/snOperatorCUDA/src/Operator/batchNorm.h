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
#include "snOperatorCUDA/src/structurs.h"

class BatchNorm final : SN_Base::OperatorBase{

public:

    BatchNorm(void* net, const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

    ~BatchNorm() = default;
                
    std::vector<std::string> Do(const SN_Base::operationParam&, const std::vector<OperatorBase*>& neighbOpr) override;

    bool setBatchNorm(const SN_Base::batchNorm& bn) override;
    
    SN_Base::batchNorm BatchNorm::getBatchNorm()const override;

private: 

    std::map<std::string, SN_Base::snFloat*> auxGPUParams_;       ///< aux data 
    mutable std::map<std::string, std::vector<SN_Base::snFloat>> auxCPUParams_;

    SN_Base::snSize inSzMem_;                                         ///< insz mem


    void updateConfig(bool isLern, const SN_Base::snSize& newsz);
};
