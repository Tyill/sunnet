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


/// прямой проход
bool fwdPooling(int type,   ///< тип: max, avr..
              size_t kernel,   ///< размер маски
       SN_Base::snSize insz,   ///< вход значения размер 
    SN_Base::snFloat* input,   ///< вход значения
      SN_Base::snSize outsz,   ///< выход значения размер 
   SN_Base::snFloat* output,   ///< выход значения
          size_t* outputInx);  ///< выход значения индекс ненулевого элемента

/// обратный проход
bool bwdPooling(int type,   ///< тип: max, avr..
              size_t kernel,   ///< размер маски
      SN_Base::snSize outsz,   ///< выход значения размер 
          size_t* outputInx,   ///< выход значения индекс ненулевого элемента
   SN_Base::snFloat* gradIn,   ///< входной градиент
       SN_Base::snSize insz,   ///< вход значения размер 
  SN_Base::snFloat* gradOut);  ///< выходной градиент


/// объединяющий слой
class Pooling : SN_Base::OperatorBase{

public:

    Pooling(void* net, const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

    ~Pooling() = default;

    std::vector<std::string> Do(const SN_Base::operationParam&, const std::vector<OperatorBase*>& neighbOpr) override;
    
        
private:
        
    size_t kernel_ = 2;                                               ///< размер
    
    poolType poolType_ = poolType::max;                               ///< тип
                                                                      
    SN_Base::snSize inSzMem_;                                         ///< размер вх данных

    std::vector<size_t> outInx_;                                      ///< индекс выбран эл-та (если maxPool)

    SN_Base::snSize inDataExpSz_;                                     ///< размер вх данных
    std::vector<SN_Base::snFloat> inDataExp_;                         ///< вход данные расширен

    size_t paddingH_ = 0, paddingW_ = 0;                              ///< доп отступ по краям для свертки
    bool isPadding_ = false;

    std::map<std::string, std::vector<SN_Base::snFloat>> auxParams_;  ///< вспом данные для расчета

    void load(std::map<std::string, std::string>& prms);
        
    void updateConfig(const SN_Base::snSize& newSz);

    void paddingOffs(bool in2out, const SN_Base::snSize& insz, SN_Base::snFloat* in, SN_Base::snFloat* out);

    void forward(SN_Base::Tensor* inTns);
    void backward(SN_Base::Tensor* inTns, const SN_Base::operationParam& operPrm);


 
};