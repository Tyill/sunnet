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

#include <mutex>
#include "snBase/snBase.h"
#include "skynet/skynet.h"
#include "snOperator/snOperator.h"
#include "src/threadPool.h"

namespace SN_Eng{

    class SNEngine{

    public:
        SNEngine(SN_Base::Net&, SN_API::snStatusCBack, SN_API::snUData);
        
        ~SNEngine();
    
        /// прямой проход
        bool forward(const SN_Base::operationParam&);

        /// обратный проход
        bool backward(const SN_Base::operationParam&);
        
    private:
        
        struct ndState{
            bool isWasRun = false;                             ///< был запущен на предыд итерации
            std::vector<std::string> selectNextNodes;          ///< были выбраны на предыд итерации след узлы (множест число, тк мбыть разделение на неск параллель нитей)
        };

        std::map<std::string, ndState> ndStates_;              ///< состояния узлов. ключ - название узла
        std::map<std::string, SN_Base::Node> nodes_;           ///< узлы сети. ключ - название узла
        std::map<std::string, SN_Base::OperatorBase*> operats_;///< все операторы. ключ - название узла
                
        SN_API::snUData udata_ = nullptr;
        SN_API::snStatusCBack stsCBack_ = nullptr;

        ThreadPool* thrPoolForward_ = nullptr, *thrPoolBackward_ = nullptr;
        bool fWorkEnd_ = false;                       ///< закрытие всех потоков
    
        SN_Base::operationParam operParam_;            ///< параметры тек итерации

        void statusMess(const std::string&);
            
        /// создание потоков
        void createThreads(std::map<std::string, SN_Base::Node>& nodes);

        /// выполнение оператора при движении вперед 
        void actionForward(std::map<std::string, SN_Base::Node>& nodes, const std::string& nname);

        /// выполнение оператора при движении назад
        void actionBackward(std::map<std::string, SN_Base::Node>& nodes, const std::string& nname);

        /// сброс готовности старта для след-х узлов
        void resetPreStartNode(std::map<std::string, SN_Base::Node>& nodes, const std::string& nname);

        /// выбор след узла при движении вперед
        std::string selectNextForward(std::map<std::string, SN_Base::Node>& nodes, const std::string& nname, std::string& nnamemem);
        
        /// выбор след узла при движении назад
        std::string selectNextBackward(std::map<std::string, SN_Base::Node>& nodes, const std::string& nname, std::string& nnamemem);

        /// рабочий поток для каждой нити операторов
        void operatorThreadForward(std::string node);
        void operatorThreadBackward(std::string node);

    };
}
