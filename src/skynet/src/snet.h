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
#include "snEngine/snEngine.h"
#include "skynet/skynet.h"
#include "Lib/rapidjson/document.h"

class SNet
{
public:
    SNet(const char* jnNet, char* out_err /*sz 256*/, SN_API::snStatusCBack = nullptr, SN_API::snUData = nullptr);
    ~SNet();
        
    void getLastErrorStr(char* out_err);

    bool training(SN_Base::snFloat lr, const SN_Base::snSize& isz, const SN_Base::snFloat* iLayer,
        const SN_Base::snSize& osz, SN_Base::snFloat* outData, const SN_Base::snFloat* targetData, SN_Base::snFloat* outAccurate);

    bool forward(bool isLern, const SN_Base::snSize& isz, const SN_Base::snFloat* iLayer, const SN_Base::snSize& osz, SN_Base::snFloat* outData);
    
    bool backward(SN_Base::snFloat lr, const SN_Base::snSize& gsz, const SN_Base::snFloat* grad);

    bool setWeightNode(const char* nodeName, const SN_Base::snSize& wsz, const SN_Base::snFloat* wData);

    bool getWeightNode(const char* nodeName, SN_Base::snSize& wsz, SN_Base::snFloat** wData);
        
    bool setBatchNormNode(const char* nodeName, const SN_Base::batchNorm&);

    bool getBatchNormNode(const char* nodeName, SN_Base::batchNorm&);

    bool setInputNode(const char* nodeName, const SN_Base::snSize& isz, const SN_Base::snFloat* inData);

    bool getOutputNode(const char* nodeName, SN_Base::snSize& osz, SN_Base::snFloat** outData);

    bool setGradientNode(const char* nodeName, const SN_Base::snSize& gsz, const SN_Base::snFloat* gData);

    bool getGradientNode(const char* nodeName, SN_Base::snSize& gsz, SN_Base::snFloat** gData);

    bool setParamNode(const char* nodeName, const char* jnParam);

    bool getParamNode(const char* nodeName, char** jnParam);

    bool getArchitecNet(char** jnArchitecNet);

    bool snAddUserCallBack(const char* ucbName, SN_API::snUserCBack, SN_API::snUData);
   
    bool saveAllWeightToFile(const char* filePath);

    bool loadAllWeightFromFile(const char* filePath);

    void statusMess(const std::string&);

    void userCBack(const std::string& cbname, const std::string& node,
        bool fwBw, const SN_Base::snSize& insz, SN_Base::snFloat* in, SN_Base::snSize& outsz, SN_Base::snFloat** out);
    
private:

    SN_Eng::SNEngine* engine_ = nullptr;                        ///< driver 
        
    std::map<std::string, SN_Base::Node> nodes_;                ///< all nodes of net
    std::map<std::string, SN_Base::OperatorBase*> operats_;     ///< all operators. key - name node

    std::mutex mtxCmn_;
                
    std::map<std::string, SN_Base::Tensor*> weight_;            ///< weight node's. key - name node
    std::map<std::string, SN_Base::Tensor*> inData_;            ///< input data node's. key - name node
    std::map<std::string, SN_Base::Tensor*> gradData_;          ///< grad data node's. key - name node

    /// sts callBack
    std::map<std::string, std::pair<SN_API::snUserCBack, SN_API::snUData>> userCBack_;

    SN_Base::operationParam operPrm_;                           ///< param current operation

    rapidjson::Document jnNet_;                                 ///< architec of net

    bool isBeginNet_ = false, isEndNet_ = false;

    SN_API::snUData udata_ = nullptr;
    SN_API::snStatusCBack stsCBack_ = nullptr;

    std::string lastError_;
   
  
    bool jnParseNet(const std::string& branchJSON, SN_Base::Net& out_net, std::string& out_err);
  
    bool checkCrossRef(std::map<std::string, SN_Base::Node>& nodes, std::string& err);
 
    bool createNet(SN_Base::Net& inout_net, std::string& out_err);
 
    SN_Base::snFloat calcAccurate(SN_Base::Tensor* targetTens, SN_Base::Tensor* outTens);
};
