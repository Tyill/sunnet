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
#include "../stdafx.h"
#include "Output.h"

using namespace std;
using namespace SN_Base;



/// конец сети - оператор заглушка - ничего не должен делать!
Output::Output(const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(name, node, prms){
        
}

/// выполнить расчет
std::vector<std::string> Output::Do(const learningParam& lernPrm, const std::vector<OperatorBase*>& neighbOpr){
        
    if (neighbOpr.size() == 1){
        if (lernPrm.action == snAction::forward)
            baseOut_ = neighbOpr[0]->getOutput();
    }
    else{
        if (lernPrm.action == snAction::forward){

            inFwTns_ = *neighbOpr[0]->getOutput();

            size_t sz = neighbOpr.size();
            for (size_t i = 1; i < sz; ++i)
                inFwTns_ += *neighbOpr[i]->getOutput();

            baseOut_ = &inFwTns_;
        }
    }

    return std::vector<std::string>();
}


