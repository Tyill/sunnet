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
    

/// сверточный слой
class Convolution : SN_Base::OperatorBase{

public:

    Convolution(void* net, const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

    ~Convolution();

    std::vector<std::string> Do(const SN_Base::operationParam&, const std::vector<OperatorBase*>& neighbOpr) override;
        
    bool setInternPrm(std::map<std::string, std::string>& prms) override;
    
    bool setBatchNorm(const SN_Base::batchNorm& bn) override;

private:
        
    size_t kernel_ = 10;                                        ///< кол-во вых слоев свертки
    size_t fWidth_ = 3;                                         ///< длина слоя свертки
    size_t fHeight_ = 3;                                        ///< высота слоя свертки    
    size_t stride_ = 1;                                         ///< шаг перемещения свертки
    size_t paddingSet_ = 0, paddingH_ = 0, paddingW_ = 0;       ///< доп отступ по краям для свертки

    bool isPaddingSame_ = false;

    activeType activeType_ = activeType::relu;                  ///< тип ф-ии активации
    optimizerType optimizerType_ = optimizerType::adam;         ///< тип оптимизатора весов
    weightInitType weightInitType_ = weightInitType::he;        ///< тип инициализации весов
    batchNormType batchNormType_ = batchNormType::none;         ///< тип batchNorm 
    SN_Base::snSize inSzMem_;                                   ///< размер вх данных
    SN_Base::snSize inDataExpSz_;                               ///< размер вх данных
    std::vector<SN_Base::snFloat> inDataExp_;                   ///< вход данные расширен
          
    bool isFreeze_ = false;                                     ///< не менять веса

    calcMode calcMode_ = calcMode::CPU;                         ///< режим расчета

    SN_Base::snFloat opt_decayMomentDW_ = 0.9F,                 ///< оптимизация изм весов
                     opt_decayMomentWGr_ = 0.99F,
                     opt_lmbRegular_ = 0.001F;

    std::map<std::string, std::vector<SN_Base::snFloat>> auxParams_;  ///< вспом данные для расчета
    std::map<std::string, SN_Base::snFloat*> gpuParams_;              ///< вспом для CUDA и OpenCL

    void load(std::map<std::string, std::string>& prms);

    void updateConfig(const SN_Base::snSize& newSz);
    
    void paddingOffs(bool in2out, const SN_Base::snSize& insz, SN_Base::snFloat* in, SN_Base::snFloat* out);

    void calcBatchNorm(bool fwBw, bool isLern, const SN_Base::snSize& insz, SN_Base::snFloat* in, SN_Base::snFloat* out, SN_Base::batchNorm& prm);
       
    void forward(SN_Base::Tensor* inTns, const SN_Base::operationParam& operPrm);
    void backward(SN_Base::Tensor* inTns, const SN_Base::operationParam& operPrm);
       
    /// CPU ///////////////////////////

    /// прямой проход
    void forwardCPU(size_t kernel,   ///< колво вых слоев
        size_t fWidth,               ///< ширина маски
        size_t fHeight,              ///< высота маски
        size_t stride,               ///< шаг движения маски
        SN_Base::snFloat* weight,    ///< веса
        SN_Base::snSize insz,        ///< вход значения размер 
        SN_Base::snFloat* input,     ///< вход значения
        SN_Base::snSize outsz,       ///< выход значения размер 
        SN_Base::snFloat* output);   ///< выход знач (скрытых нейронов) для след слоя

    /// обратный проход. Расчет град-в и весов
    void backwardCPU_GW(size_t kernel, ///< колво вых слоев
        size_t fWidth,                 ///< ширина маски
        size_t fHeight,                ///< высота маски
        size_t stride,                 ///< шаг движения маски
        SN_Base::snFloat* weight,      ///< веса
        SN_Base::snSize insz,          ///< вход значения размер 
        SN_Base::snFloat* input,       ///< вход значения 
        SN_Base::snSize outsz,         ///< выход значения размер 
        SN_Base::snFloat* gradIn,      ///< вход градиент ошибки с пред слоя
        SN_Base::snFloat* gradOut,     ///< выход градиент ошибки для след слоя
        SN_Base::snFloat* dWeightOut); ///< дельта изменения весов

    /// обратный проход. Расчет град-в
    void backwardCPU_G(size_t kernel,  ///< колво вых слоев
        size_t fWidth,                 ///< ширина маски
        size_t fHeight,                ///< высота маски
        size_t stride,                 ///< шаг движения маски
        SN_Base::snFloat* weight,      ///< веса
        SN_Base::snSize insz,          ///< вход значения размер 
        SN_Base::snFloat* input,       ///< вход значения 
        SN_Base::snSize outsz,         ///< выход значения размер 
        SN_Base::snFloat* gradIn,      ///< вход градиент ошибки с пред слоя
        SN_Base::snFloat* gradOut);    ///< выход градиент ошибки для след слоя



    /// CUDA ///////////////////////////
    
    /// иниц вспом параметров CUDA          
    void iniParamCUDA(SN_Base::snSize insz, SN_Base::snSize outsz, size_t fWidth, size_t fHeight, std::map<std::string, SN_Base::snFloat*>& gpuPrm);

    /// освоб вспом параметров CUDA          
    void freeParamCUDA(std::map<std::string, SN_Base::snFloat*>& gpuPrm);

    /// прямой проход
    void forwardCUDA(size_t kernel,   ///< колво вых слоев
        size_t fWidth,               ///< ширина маски
        size_t fHeight,              ///< высота маски
        size_t stride,               ///< шаг движения маски
        SN_Base::snFloat* weight,    ///< веса
        SN_Base::snSize insz,        ///< вход значения размер 
        SN_Base::snFloat* input,     ///< вход значения
        SN_Base::snSize outsz,       ///< выход значения размер 
        SN_Base::snFloat* output,    ///< выход знач (скрытых нейронов) для след слоя
        std::map<std::string, SN_Base::snFloat*>&); ///< вспом  

    /// обратный проход. Расчет град-в и весов
    void backwardCUDA_GW(size_t kernel, ///< колво вых слоев
        size_t fWidth,                 ///< ширина маски
        size_t fHeight,                ///< высота маски
        size_t stride,                 ///< шаг движения маски
        SN_Base::snFloat* weight,      ///< веса
        SN_Base::snSize insz,          ///< вход значения размер 
        SN_Base::snFloat* input,       ///< вход значения 
        SN_Base::snSize outsz,         ///< выход значения размер 
        SN_Base::snFloat* gradIn,      ///< вход градиент ошибки с пред слоя
        SN_Base::snFloat* gradOut,     ///< выход градиент ошибки для след слоя
        SN_Base::snFloat* dWeightOut,  ///< дельта изменения весов
        std::map<std::string, SN_Base::snFloat*>&);  ///< вспом 

    /// обратный проход. Расчет град-в
    void backwardCUDA_G(size_t kernel,  ///< колво вых слоев
        size_t fWidth,                 ///< ширина маски
        size_t fHeight,                ///< высота маски
        size_t stride,                 ///< шаг движения маски
        SN_Base::snFloat* weight,      ///< веса
        SN_Base::snSize insz,          ///< вход значения размер 
        SN_Base::snFloat* input,       ///< вход значения 
        SN_Base::snSize outsz,         ///< выход значения размер 
        SN_Base::snFloat* gradIn,      ///< вход градиент ошибки с пред слоя
        SN_Base::snFloat* gradOut,     ///< выход градиент ошибки для след слоя
        std::map<std::string, SN_Base::snFloat*>&);  ///< вспом 
};