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
#include"snOperator/src/structurs.h"
#include"snOperator/src/mathFunctions.h"


/// pooling layer
class Pooling final : SN_Base::OperatorBase{

public:

    Pooling(void* net, const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

    ~Pooling();

    std::vector<std::string> Do(const SN_Base::operationParam&, const std::vector<OperatorBase*>& neighbOpr) override;
    
        
private:
    
    struct poolParams{
        poolType poolType = poolType::max;                            ///< type
        size_t kernel = 2;                                            ///< mask size
        size_t stride = 2;                                            ///< step mask
        size_t paddingH = 0, paddingW = 0;
    };

    poolParams poolPrms_;
                                                                      
    SN_Base::snSize inSzMem_;                                         ///< input size mem
    SN_Base::snSize inDataExpSz_;                                     ///< input size expand
    std::vector<size_t> outInx_;                                      ///< index select elem

    SN_Base::Tensor inTnsExp_;
    SN_Base::Tensor gradOutExp_;

                                
    bool isPadding_ = false;
    
    bool gpuClearMem_ = false;                                        ///< freee gpu mem

    uint32_t gpuDeviceId_ = 0;                                        ///< gpu id

    calcMode calcMode_ = calcMode::CPU;                               ///< calc mode


    std::map<std::string, std::vector<SN_Base::snFloat>> auxParams_;  ///< aux data
    void* gpuParams_ = nullptr;                                       ///< gpu data

    void load(std::map<std::string, std::string>& prms);
        
    void updateConfig(const SN_Base::snSize& newSz);
  
    void forward(SN_Base::Tensor* inTns, const SN_Base::operationParam& operPrm);
    void backward(SN_Base::Tensor* inTns, const SN_Base::operationParam& operPrm);

    /// CPU ///////////////////////////

    void forwardCPU(const poolParams& poolPrms,
        const SN_Base::snSize& insz,   
        SN_Base::snFloat* input,       
        const SN_Base::snSize& outsz,  
        SN_Base::snFloat* output,      
        size_t* outputInx);            

    void backwardCPU(const poolParams& poolPrms,
        const SN_Base::snSize& outsz, 
        size_t* outputInx,            
        SN_Base::snFloat* gradIn,     
        const SN_Base::snSize& insz,  
        SN_Base::snFloat* gradOut);  


    /// CUDA ///////////////////////////

    void iniParamCUDA(const SN_Base::snSize& insz, const SN_Base::snSize& outsz, const poolParams&, void** gpuPrm);

    void freeParamCUDA(void* gpuPrm);

    void forwardCUDA(const poolParams& poolPrms,
        const SN_Base::snSize& insz,    
        SN_Base::snFloat* input,        
        const SN_Base::snSize& outsz,   
        SN_Base::snFloat* output,       
        size_t* outputInx,              
        void* gpuParams);

    /// обратный проход CUDA
    void backwardCUDA(const poolParams& poolPrms,
        const SN_Base::snSize& outsz,   
        size_t* outputInx,       
        SN_Base::snFloat* output,
        SN_Base::snFloat* gradIn,       
        const SN_Base::snSize& insz,
        SN_Base::snFloat* input,
        SN_Base::snFloat* gradOut,      
        void* gpuParams);


    /// OpenCL ///////////////////////////

    void iniParamOCL(const SN_Base::snSize& insz, const SN_Base::snSize& outsz, const poolParams&, void** gpuPrm);

    void freeParamOCL(void* gpuPrm);

    void forwardOCL(const poolParams& poolPrms,
        const SN_Base::snSize& insz,      
        SN_Base::snFloat* input,          
        const SN_Base::snSize& outsz,     
        SN_Base::snFloat* output,         
        size_t* outputInx,                
        void* gpuParams);

    void backwardOCL(const poolParams& poolPrms,
        const SN_Base::snSize& outsz,    
        size_t* outputInx,               
        SN_Base::snFloat* gradIn,        
        const SN_Base::snSize& insz,     
        SN_Base::snFloat* gradOut,       
        void* gpuParams);
 
};