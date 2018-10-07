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


/// pooling layer
class Pooling : SN_Base::OperatorBase{

public:

    Pooling(void* net, const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

    ~Pooling();

    std::vector<std::string> Do(const SN_Base::operationParam&, const std::vector<OperatorBase*>& neighbOpr) override;
    
        
private:
        
    size_t kernel_ = 2;                                               ///< mask size
    
    poolType poolType_ = poolType::max;                               ///< type
                                                                      
    SN_Base::snSize inSzMem_;                                         ///< input size mem
    SN_Base::snSize inDataExpSz_;                                     ///< input size expand
    std::vector<size_t> outInx_;                                      ///< index select elem

    SN_Base::Tensor inTnsExp_;
    SN_Base::Tensor gradOutExp_;

    size_t paddingH_ = 0, paddingW_ = 0;                              
    bool isPadding_ = false;
    
    bool gpuClearMem_ = false;                                        ///< freee gpu mem

    uint32_t gpuDeviceId_ = 0;                                        ///< gpu id

    calcMode calcMode_ = calcMode::CPU;                               ///< calc mode


    std::map<std::string, std::vector<SN_Base::snFloat>> auxParams_;  ///< aux data
    std::map<std::string, void*> gpuParams_;                          

    void load(std::map<std::string, std::string>& prms);
        
    void updateConfig(const SN_Base::snSize& newSz);
      
    void paddingOffs(bool in2out, const SN_Base::snSize& insz, SN_Base::snFloat* in, SN_Base::snFloat* out);

    void forward(SN_Base::Tensor* inTns, const SN_Base::operationParam& operPrm);
    void backward(SN_Base::Tensor* inTns, const SN_Base::operationParam& operPrm);

    /// CPU ///////////////////////////

    void forwardCPU(poolType type,     
        size_t kernel,                 
        const SN_Base::snSize& insz,   
        SN_Base::snFloat* input,       
        const SN_Base::snSize& outsz,  
        SN_Base::snFloat* output,      
        size_t* outputInx);            

   void backwardCPU(poolType type,    
        size_t kernel,                
        const SN_Base::snSize& outsz, 
        size_t* outputInx,            
        SN_Base::snFloat* gradIn,     
        const SN_Base::snSize& insz,  
        SN_Base::snFloat* gradOut);  


    /// CUDA ///////////////////////////

    void iniParamCUDA(const SN_Base::snSize& insz, const SN_Base::snSize& outsz, size_t kernel, std::map<std::string, void*>& gpuPrm);

    void freeParamCUDA(std::map<std::string, void*>& gpuPrm);

    void forwardCUDA(poolType type,     
        size_t kernel,                  
        const SN_Base::snSize& insz,    
        SN_Base::snFloat* input,        
        const SN_Base::snSize& outsz,   
        SN_Base::snFloat* output,       
        size_t* outputInx,              
        std::map<std::string, void*>& gpuParams);

    /// обратный проход CUDA
    void backwardCUDA(poolType type,    
        size_t kernel,                  
        const SN_Base::snSize& outsz,   
        size_t* outputInx,       
        SN_Base::snFloat* output,
        SN_Base::snFloat* gradIn,       
        const SN_Base::snSize& insz,
        SN_Base::snFloat* input,
        SN_Base::snFloat* gradOut,      
        std::map<std::string, void*>& gpuParams);


    /// OpenCL ///////////////////////////

    void iniParamOCL(const SN_Base::snSize& insz, const SN_Base::snSize& outsz, size_t kernel, std::map<std::string, void*>& gpuPrm);

    void freeParamOCL(std::map<std::string, void*>& gpuPrm);

    void forwardOCL(poolType type,        
        size_t kernel,                    
        const SN_Base::snSize& insz,      
        SN_Base::snFloat* input,          
        const SN_Base::snSize& outsz,     
        SN_Base::snFloat* output,         
        size_t* outputInx,                
        std::map<std::string, void*>& gpuParams);

   void backwardOCL(poolType type,       
        size_t kernel,                   
        const SN_Base::snSize& outsz,    
        size_t* outputInx,               
        SN_Base::snFloat* gradIn,        
        const SN_Base::snSize& insz,     
        SN_Base::snFloat* gradOut,       
        std::map<std::string, void*>& gpuParams);
 
};