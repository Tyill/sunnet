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
#include"SNOperator/src/structurs.h"
#include"SNOperator/src/mathFunctions.h"

/// convolution layer
class Convolution : SN_Base::OperatorBase{

public:

    Convolution(void* net, const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

    ~Convolution();

    std::vector<std::string> Do(const SN_Base::operationParam&, const std::vector<OperatorBase*>& neighbOpr) override;
        
    bool setInternPrm(std::map<std::string, std::string>& prms) override;
    
    bool setBatchNorm(const SN_Base::batchNorm& bn) override;

private:
    
    struct convParams{
        size_t kernel = 10;                                      ///< number of output convolution layers
        size_t fWidth = 3;                                       ///< width mask
        size_t fHeight = 3;                                      ///< height mask 
        size_t dilate = 1;                                       ///< expansion mask
        size_t stride = 1;                                       ///< step mask
        size_t paddingSet = 0, paddingH = 0, paddingW = 0;       ///< padding layer
    };

    convParams convPrms_;
    bool isPaddingSame_ = false;

    activeType activeType_ = activeType::relu;                  ///< active type
    optimizerType optimizerType_ = optimizerType::adam;         ///< optimizer type
    weightInitType weightInitType_ = weightInitType::he;        ///< init weight type
    batchNormType batchNormType_ = batchNormType::none;         ///< batchNorm 
    SN_Base::snSize inSzMem_;                                   ///< insz mem
    SN_Base::snSize inDataExpSz_;                               ///< insz expansion
   
    SN_Base::Tensor inTnsExp_;
    SN_Base::Tensor gradOutExp_;
   
    bool isFreeze_ = false;                                     ///< not change weight
    bool gpuClearMem_ = false;                                  ///< clear mem GPU

    uint32_t gpuDeviceId_ = 0;                                  ///< gpu id

    calcMode calcMode_ = calcMode::CPU;                         ///< calc mode

    SN_Base::snFloat dropOut_ = 0.F;                            ///< random off out

    SN_Base::snFloat opt_decayMomentDW_ = 0.9F,                 ///< optimiz weight
                     opt_decayMomentWGr_ = 0.99F,
                     opt_lmbRegular_ = 0.001F;

    std::map<std::string, std::vector<SN_Base::snFloat>> auxParams_;  ///< aux data 
    std::map<std::string, void*> gpuParams_;                         
       

    void load(std::map<std::string, std::string>& prms);

    void updateConfig(const SN_Base::snSize& newSz, SN_Base::snSize& expSz);
    
    void paddingOffs(bool in2out, const SN_Base::snSize& insz, SN_Base::snFloat* in, SN_Base::snFloat* out);
      
    void calcBatchNorm(bool fwBw, bool isLern, const SN_Base::snSize& insz, SN_Base::snFloat* in, SN_Base::snFloat* out, SN_Base::batchNorm& prm);
       
    void forward(SN_Base::Tensor* inTns, const SN_Base::operationParam& operPrm);
    void backward(SN_Base::Tensor* inTns, const SN_Base::operationParam& operPrm);
       
    /// CPU ///////////////////////////

    void forwardCPU(const convParams&, 
        SN_Base::snFloat* weight,      
        const SN_Base::snSize& insz,   
        SN_Base::snFloat* input,       
        const SN_Base::snSize& outsz,  
        SN_Base::snFloat* output);     

    // calc grad and weight
    void backwardCPU_GW(const convParams&,
        SN_Base::snFloat* weight,      
        const SN_Base::snSize& insz,   
        SN_Base::snFloat* input,       
        const SN_Base::snSize& outsz,  
        SN_Base::snFloat* gradIn,      
        SN_Base::snFloat* gradOut,     
        SN_Base::snFloat* dWeightOut); 

    // calc grad
    void backwardCPU_G(const convParams&,
        SN_Base::snFloat* weight,     
        const SN_Base::snSize& insz,  
        const SN_Base::snSize& outsz, 
        SN_Base::snFloat* gradIn,     
        SN_Base::snFloat* gradOut);   



    /// CUDA ///////////////////////////

    /// init aux params
    void iniParamCUDA(const SN_Base::snSize& insz, const SN_Base::snSize& outsz,
        const convParams&, std::map<std::string, void*>& gpuPrm);
   
    /// free aux params
    void freeParamCUDA(std::map<std::string, void*>& gpuPrm);

    void forwardCUDA(const convParams&,
        SN_Base::snFloat* weight,      
        const SN_Base::snSize& insz,   
        SN_Base::snFloat* input,       
        const SN_Base::snSize& outsz,  
        SN_Base::snFloat* output,      
        std::map<std::string, void*>&);

    /// calc grad and weight
    void backwardCUDA_GW(const convParams&,
        SN_Base::snFloat* weight,      
        const SN_Base::snSize& insz,   
        SN_Base::snFloat* input,       
        const SN_Base::snSize& outsz,  
        SN_Base::snFloat* gradIn,      
        SN_Base::snFloat* gradOut,     
        SN_Base::snFloat* dWeightOut,  
        std::map<std::string, void*>&);

    /// calc grad
    void backwardCUDA_G(const convParams&,
        SN_Base::snFloat* weight,      
        const SN_Base::snSize& insz,   
        const SN_Base::snSize& outsz,  
        SN_Base::snFloat* gradIn,      
        SN_Base::snFloat* gradOut,     
        std::map<std::string, void*>&);


    /// OpenCL ///////////////////////////

    /// init aux params OpenCL          
    void iniParamOCL(const SN_Base::snSize& insz, const SN_Base::snSize& outsz,
        const convParams&, std::map<std::string, void*>& gpuPrm);

    /// free aux params OpenCL          
    void freeParamOCL(std::map<std::string, void*>& gpuPrm);

    void forwardOCL(const convParams&,
        SN_Base::snFloat* weight,      
        const SN_Base::snSize& insz,   
        SN_Base::snFloat* input,       
        const SN_Base::snSize& outsz,  
        SN_Base::snFloat* output,      
        std::map<std::string, void*>&);

    /// calc grad and weight
    void backwardOCL_GW(const convParams&,
        SN_Base::snFloat* weight,      
        const SN_Base::snSize& insz,   
        SN_Base::snFloat* input,       
        const SN_Base::snSize& outsz,  
        SN_Base::snFloat* gradIn,      
        SN_Base::snFloat* gradOut,     
        SN_Base::snFloat* dWeightOut,  
        std::map<std::string, void*>&); 

    /// calc grad
    void backwardOCL_G(const convParams&,
        SN_Base::snFloat* weight,      
        const SN_Base::snSize& insz,   
        const SN_Base::snSize& outsz,  
        SN_Base::snFloat* gradIn,      
        SN_Base::snFloat* gradOut,     
        std::map<std::string, void*>&);
};