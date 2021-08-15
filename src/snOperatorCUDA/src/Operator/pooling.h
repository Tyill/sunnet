//
// sunnet project
// Copyright (C) 2018 by Contributors <https://github.com/Tyill/sunnet>
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
#include"snOperatorCUDA/src/structurs.h"

/// pooling layer
class Pooling final : SN_Base::OperatorBase{

public:

    Pooling(void* net, const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

    ~Pooling();

    std::vector<std::string> Do(const SN_Base::operationParam&, const std::vector<OperatorBase*>& neighbOpr) override;
    
        
private:
    
    struct poolParams{
        poolType type = poolType::max;                                ///< type
        size_t kernel = 2;                                            ///< mask size
        size_t stride = 2;                                            ///< step mask
        size_t paddingH = 0, paddingW = 0;
    };

    poolParams poolPrms_;
                                                                      
    SN_Base::snSize inSzMem_;                                         ///< input size mem
  
    const SN_Base::Tensor* inputMem_ = nullptr;

                                
    bool isPadding_ = false;
      
    uint32_t gpuDeviceId_ = 0;                                        ///< gpu id

  
    std::map<std::string, std::vector<SN_Base::snFloat>> auxParams_;  ///< aux data
    void* gpuParams_ = nullptr;                                       ///< gpu data

    void load(std::map<std::string, std::string>& prms);
        
    void updateConfig(bool isLern, const SN_Base::snSize& newSz);
  
    void forward(const SN_Base::Tensor& inTns, const SN_Base::operationParam& operPrm);
    void backward(const SN_Base::Tensor& inTns, const SN_Base::operationParam& operPrm);

   
    /// CUDA ///////////////////////////

    void iniParamCUDA(bool isLern, const SN_Base::snSize& insz, const SN_Base::snSize& outsz, const poolParams&, void** gpuPrm);

    void freeParamCUDA(void* gpuPrm);

    void forwardCUDA(const poolParams& poolPrms,
        const SN_Base::snSize& insz,    
        const SN_Base::snFloat* input,
        const SN_Base::snSize& outsz,   
        SN_Base::snFloat* output,       
        void* gpuParams);

    /// РѕР±СЂР°С‚РЅС‹Р№ РїСЂРѕС…РѕРґ CUDA
    void backwardCUDA(const poolParams& poolPrms,
        const SN_Base::snSize& outsz,  
        const SN_Base::snFloat* output,
        const SN_Base::snFloat* gradIn,
        const SN_Base::snSize& insz,
        const SN_Base::snFloat* input,
        SN_Base::snFloat* gradOut,      
        void* gpuParams);
};