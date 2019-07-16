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

#include <algorithm>
#include <iterator>
#include <fstream>
#include "stdafx.h"
#include "snAux/auxFunc.h"
#include "snBase/snBase.h"
#include "snet.h"
#include "snOperatorCPU/snOperator.h"

using namespace std;
using namespace SN_Base;

void g_statusMess(SN_Base::OperatorBase* opr, const std::string& mess){

    (static_cast<SNet*>(opr->Net))->statusMess(mess);
}

void g_userCBack(SN_Base::OperatorBase* opr, const std::string& cbname, const std::string& node,
    bool fwBw, const snSize& insz, snFloat* in, snSize& outsz, snFloat** out){

    (static_cast<SNet*>(opr->Net))->userCBack(cbname, node, fwBw, insz, in, outsz, out);
}

void SNet::statusMess(const string& mess){

    lastError_ = mess;
    if (stsCBack_) stsCBack_(mess.c_str(), udata_);
}

void SNet::getLastErrorStr(char* out_err){

    if (out_err)
        strcpy(out_err, lastError_.c_str());
}

void SNet::userCBack(const std::string& cbname, const std::string& node, bool fwBw, const snSize& insz, snFloat* in, snSize& outsz, snFloat** out){

    if (userCBack_.find(cbname) != userCBack_.end()){
     
        SN_API::snLSize inlsz(insz.w, insz.h, insz.d, insz.n);
        SN_API::snLSize outlsz;

        userCBack_[cbname].first(cbname.c_str(), node.c_str(), fwBw, inlsz, in, &outlsz, out, userCBack_[cbname].second);

        outsz.w = outlsz.w;
        outsz.h = outlsz.h;
        outsz.d = outlsz.ch;
        outsz.n = outlsz.bsz;
    }
    else
        statusMess("userCBack error: not found cbname '" + cbname + "'");
}

// checking links between nodes
bool SNet::checkCrossRef(std::map<std::string, SN_Base::Node>& nodes, std::string& err){
        
    for (auto& n : nodes){

        // checking for nodes
        if ((n.second.name == "") || (n.second.oprName == "")){
            err = "Error createNet: node '" + n.first + "' - not found";
            return false;
        }

        for (auto& next : n.second.nextNodes){

            if (nodes.find(next) == nodes.end()){
                err = "Error createNet: node '" + n.first + "' - not found next node '" + next + "'";
                statusMess(err);
                return false;
            }
        }

        for (auto& prev : n.second.prevNodes){

            if (nodes.find(prev) == nodes.end()){
                err = "Error createNet: node '" + n.first + "' - not found prev node '" + prev + "'";
                statusMess(err);
                return false;
            }
        }

        for (auto& next : n.second.nextNodes){

            auto& prevNodes = nodes[next].prevNodes;

            if (find(prevNodes.begin(), prevNodes.end(), n.first) == prevNodes.end()){

                err = "Error createNet: node '" + next + "' - not found prevNode '" + n.first + "'";
                statusMess(err);
                return false;
            }
        }

        for (auto& prev : n.second.prevNodes){

            auto& nextNodes = nodes[prev].nextNodes;

            if (find(nextNodes.begin(), nextNodes.end(), n.first) == nextNodes.end()){
                err = "Error createNet: node '" + prev + "' - not found nextNode '" + n.first + "'";
                statusMess(err);
                return false;
            }
        }

        vector<string> prevNodes = n.second.prevNodes;
        while (!prevNodes.empty()){
        
            string pnd = prevNodes.back();
            prevNodes.pop_back();

            if (!nodes[pnd].prevNodes.empty()){

                for (auto n : nodes[pnd].prevNodes){
                    prevNodes.push_back(n);
                }
            }
            else if (nodes[pnd].oprName != "Input"){
                err = "Error createNet: node '" + pnd + "' - not found begin node 'Input'";
                statusMess(err);
                return false;
            }
            else continue;            
        }

        vector<string> nextNodes = n.second.nextNodes;
        while (!nextNodes.empty()){

            string nnd = nextNodes.back();
            nextNodes.pop_back();

            if (!nodes[nnd].nextNodes.empty()){

                for (auto n : nodes[nnd].nextNodes){
                    nextNodes.push_back(n);
                }
            }
            else if (nodes[nnd].oprName != "Output"){
                err = "Error createNet: node '" + nnd + "' - not found end node 'Output'";
                statusMess(err);
                return false;
            }
            else continue;
        }
    }

    return true;
}

bool SNet::createNet(Net& inout_net, std::string& out_err){
        
    // checking links between nodes
    if (!checkCrossRef(inout_net.nodes, out_err)) return false;

    for (auto& n : inout_net.nodes){

        OperatorBase* opr = SN_Opr::createOperator(this, n.second.oprName, n.first, n.second.oprPrms);

        if (!opr){
            out_err = "Error createNet: not found operator '" + n.second.oprName + "'";
            inout_net.operats.clear();
            return false;
        }
        inout_net.operats[n.first] = opr;
    }
            
    return true;
}

SNet::SNet(const char* jnNet, char* out_err /*sz 256*/,
    SN_API::snStatusCBack sts, SN_API::snUData ud) : stsCBack_(sts), udata_(ud){

    string err;  SN_Base::Net net;
    if (!jnParseNet(jnNet, net, err)){
        statusMess(err);
        strcpy(out_err, err.c_str());
        return;
    }
     
    if (!createNet(net, err)){
        statusMess(err);
        strcpy(out_err, err.c_str());
        return;
    }

    nodes_ = net.nodes;
    operats_ = net.operats;

    isBeginNet_ = operats_.find("BeginNet") != operats_.end();
    isEndNet_ = operats_.find("EndNet") != operats_.end();

    engine_ = new SN_Eng::SNEngine(net, bind(&SNet::statusMess, this, std::placeholders::_1));
}

SNet::~SNet(){

    if (engine_) delete engine_;

    for (auto o : operats_)
        SN_Opr::freeOperator(o.second, o.second->name());
}

bool SNet::training(snFloat lr, const snSize& isz, const snFloat* iLayer, const snSize& osz, snFloat* outData, const snFloat* targetData, snFloat* outAccurate){
    
    // go ahead
    if (!forward(true, isz, iLayer, osz, outData))
        return false;

    std::unique_lock<std::mutex> lk(mtxCmn_);
       
    if (!isEndNet_){
        statusMess("training error: 'EndNet' not found");
        return false;
    }
    
    // go back  
    operats_["EndNet"]->setGradient(targetData, osz);

    operPrm_.lr = lr;
    operPrm_.action = snAction::backward;
    operPrm_.isLerning = true;
    engine_->backward(operPrm_);

    // metrics
    if (outAccurate){
        auto setGrad = operats_["EndNet"]->getGradient();
        auto outTensor = operats_["EndNet"]->getOutput();
        *outAccurate = calcAccurate(setGrad, outTensor);
    }

    return true;
}

bool SNet::forward(bool isLern, const snSize& isz, const snFloat* iLayer, const snSize& osz, snFloat* outData){
    std::unique_lock<std::mutex> lk(mtxCmn_);
    
    if (!engine_){
        statusMess("forward error: net not create");
        return false;
    }
   
    if (isBeginNet_)
        operats_["BeginNet"]->setInput(iLayer, isz);

    operPrm_.action = snAction::forward;
    operPrm_.isLerning = isLern;
    engine_->forward(operPrm_);

    if (isEndNet_){
        auto tnsOut = operats_["EndNet"]->getOutput();

        auto tnsOutSz = tnsOut.size();
        if (tnsOutSz != osz){
            statusMess("forward error: tnsOutSz != osz. Must be osz: " +
                to_string(tnsOutSz.w) + " " + to_string(tnsOutSz.h) + " " + to_string(tnsOutSz.d) + " " + to_string(tnsOutSz.n));
            return false;
        }

        memcpy(outData, tnsOut.getData(), tnsOutSz.size() * sizeof(snFloat));
    }

    return true;
}

bool SNet::backward(snFloat lr, const snSize& gsz, const snFloat* gradErr){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    if (!engine_){
        statusMess("backward error: net not create");
        return false;
    }

    if (isEndNet_){
        auto outTensor = operats_["EndNet"]->getOutput();

        auto tsz = outTensor.size();
        if (tsz != gsz){
            statusMess("backward error: tnsOutSz != gsz. Must be gsz: " +
                to_string(tsz.w) + " " + to_string(tsz.h) + " " + to_string(tsz.d) + " " + to_string(tsz.n));
            return false;
        }

        operats_["EndNet"]->setGradient(gradErr, gsz);
    }

    operPrm_.lr = lr;
    operPrm_.action = snAction::backward;
    operPrm_.isLerning = true;
    engine_->backward(operPrm_);   

    return true;
}

SN_Base::snFloat SNet::calcAccurate(const Tensor& targetTens, const Tensor& outTens){

    snFloat* targetData = targetTens.getData();
    snFloat* outData = outTens.getData();
    
    size_t accCnt = 0, osz = outTens.size().size();
    for (size_t i = 0; i < osz; ++i){

        if (abs(outData[i] - targetData[i]) < 0.1)
            ++accCnt; 
    }

    return (accCnt * 1.F) / osz;
}

bool SNet::setWeightNode(const char* nodeName, const SN_Base::snSize& wsz, const SN_Base::snFloat* wData){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    if (operats_.find(nodeName) == operats_.end()){
        statusMess("SN error: '" + string(nodeName) + "' not found");
        return false;
    }
        
    operats_[nodeName]->setWeight(wData, wsz);
            
    return true;
}

bool SNet::getWeightNode(const char* nodeName, SN_Base::snSize& wsz, SN_Base::snFloat** wData){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    if (operats_.find(nodeName) == operats_.end()) {
        statusMess("SN error: '" + string(nodeName) + "' not found");
        return false;
    }
    
    auto weight = operats_[nodeName]->getWeight();

    wsz = weight.size();

    *wData = (snFloat*)realloc(*wData, wsz.size() * sizeof(snFloat));
        
    memcpy(*wData, weight.getData(), wsz.size() * sizeof(snFloat));

    return true;
}

bool SNet::setBatchNormNode(const char* nodeName, const SN_Base::batchNorm& bn){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    if (operats_.find(nodeName) == operats_.end()){
        statusMess("SN error: '" + string(nodeName) + "' not found");
        return false;
    }
      
    operats_[nodeName]->setBatchNorm(bn);

    return true;
}

bool SNet::getBatchNormNode(const char* nodeName, SN_Base::batchNorm& obn){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    if (operats_.find(nodeName) == operats_.end()){
        statusMess("SN error: '" + string(nodeName) + "' not found");
        return false;
    }

    obn = operats_[nodeName]->getBatchNorm();

    return true;
}

/// set the input data of the node (actual for additional inputs)
bool SNet::setInputNode(const char* nodeName, const SN_Base::snSize& isz, const SN_Base::snFloat* inData){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    if (operats_.find(nodeName) == operats_.end()){
        statusMess("SN error: '" + string(nodeName) + "' not found");
        return false;
    }

    if (nodes_[nodeName].oprName != "Input"){
        statusMess("SN error: '" + string(nodeName) + "' layer must be 'Input'");
        return false;
    }

    operats_[nodeName]->setInput(inData, isz);

    return true;
}

/// get the output values of the node (actual for additional outputs)
bool SNet::getOutputNode(const char* nodeName, SN_Base::snSize& osz, SN_Base::snFloat** outData){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    if (operats_.find(nodeName) == operats_.end()){
        statusMess("SN error: '" + string(nodeName) + "' not found");
        return false;
    }

    auto outTns = operats_[nodeName]->getOutput();

    osz = outTns.size();

    *outData = (snFloat*)realloc(*outData, osz.size() * sizeof(snFloat));

    memcpy(*outData, outTns.getData(), osz.size() * sizeof(snFloat));

    return true;
}

/// set the gradient of the node value (actual for additional outputs)
bool SNet::setGradientNode(const char* nodeName, const SN_Base::snSize& gsz, const SN_Base::snFloat* gData){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    if (operats_.find(nodeName) == operats_.end()){
        statusMess("SN error: '" + string(nodeName) + "' not found");
        return false;
    }

    if (nodes_[nodeName].oprName != "Output"){
        statusMess("SN error: '" + string(nodeName) + "' layer must be 'Output'");
        return false;
    }

    snSize tsz = operats_[nodeName]->getGradient().size();

    if (tsz != gsz){
        statusMess("setGradientNode error: tsz != dsz. Must be dsz: " +
            to_string(tsz.w) + " " + to_string(tsz.h) + " " + to_string(tsz.d) + " " + to_string(tsz.n));
        return false;
    }

    operats_[nodeName]->setGradient(gData, gsz);

    return true;
}

/// get the gradient of the node value (actual for additional outputs)
bool SNet::getGradientNode(const char* nodeName, SN_Base::snSize& gsz, SN_Base::snFloat** gData){
    std::unique_lock<std::mutex> lk(mtxCmn_);
    
    if (operats_.find(nodeName) == operats_.end()){
        statusMess("SN error: '" + string(nodeName) + "' not found");
        return false;
    }

    auto grad = operats_[nodeName]->getGradient();
    gsz = grad.size();

    *gData = (snFloat*)realloc(*gData, gsz.size() * sizeof(snFloat));

    memcpy(*gData, grad.getData(), gsz.size() * sizeof(snFloat));

    return true;
}

/// set callBack uses
bool SNet::snAddUserCallBack(const char* ucbName, SN_API::snUserCBack ucb, SN_API::snUData ud){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    userCBack_[ucbName] = pair<SN_API::snUserCBack, SN_API::snUData>(ucb, ud);

    return true;
}

bool SNet::saveAllWeightToFile(const char* filePath){
    
    SN_Aux::createSubDirectory(filePath);

    std::ofstream ofs;
    ofs.open(filePath, std::ios_base::out | std::ios_base::binary);

    if (ofs.good()) {

        snFloat* data = nullptr;
        snSize lSize;
        for (auto opr : operats_){

            data = opr.second->getWeight().getData();
            lSize = opr.second->getWeight().size();

            if (data){
                ofs << opr.first << " w " << lSize.w << " " << lSize.h << " " << lSize.d << endl;
                ofs.write((char*)data, lSize.w * lSize.h * lSize.d * sizeof(float));
            }

            batchNorm bn = opr.second->getBatchNorm();

            size_t sz = bn.sz.size() * sizeof(float);
            if (sz == 0) continue;

            lSize = bn.sz;
            
            ofs << opr.first << " bn mean " << lSize.w << " " << lSize.h << " " << lSize.d << endl;
            ofs.write((char*)bn.mean, sz);
            ofs << opr.first << " bn varce " << lSize.w << " " << lSize.h << " " << lSize.d << endl;
            ofs.write((char*)bn.varce, sz);
            ofs << opr.first << " bn scale " << lSize.w << " " << lSize.h << " " << lSize.d << endl;
            ofs.write((char*)bn.scale, sz);
            ofs << opr.first << " bn schift " << lSize.w << " " << lSize.h << " " << lSize.d << endl;
            ofs.write((char*)bn.schift, sz);
        }        
    }
    else {
        statusMess("SN error save weight, check file path: " + string(filePath));
        return false;
    }

    ofs.close();

    return true;

}

bool SNet::loadAllWeightFromFile(const char* filePath){

    std::ifstream ifs;
    ifs.open(filePath, std::ifstream::in | std::ifstream::binary);

    if (!ifs.good()){
        statusMess("SN error load weight, check file path: " + string(filePath));
        return false;
    }

    batchNorm bn;
    std::vector<SN_API::snFloat> dbW, dbBNMean, dbBNShift, dbBNScale, dbBNVarce;
    while (!ifs.eof()){

        string str = "";
        getline(ifs, str);

        auto opr = SN_Aux::split(str, " ");
        if (opr.empty()) continue;

        string nm = opr[0];

        bool oprExist = (operats_.find(nm) != operats_.end());
               
        if (!oprExist)
            statusMess("SN error: node '" + nm + "' not found");
               
        if (opr[1] == "w"){
        
            size_t w = stoi(opr[2]), h = stoi(opr[3]), d = stoi(opr[4]), dsz = w * h * d;
            snSize lsz(w, h, d);
            
            dbW.resize(dsz);
            ifs.read((char*)dbW.data(), dsz * sizeof(SN_API::snFloat));
            if (oprExist)
               operats_[nm]->setWeight(dbW.data(), lsz);
        }
        else if (opr[1] == "bn"){

            size_t w = stoi(opr[3]), h = stoi(opr[4]), d = stoi(opr[5]), dsz = w * h * d;
            
            bn.sz = snSize(w, h, d);

            if (opr[2] == "mean"){
                dbBNMean.resize(dsz);
                ifs.read((char*)dbBNMean.data(), dsz * sizeof(SN_API::snFloat));
                if (oprExist)
                   bn.mean = dbBNMean.data();
            }
            else if (opr[2] == "varce"){
                dbBNVarce.resize(dsz);
                ifs.read((char*)dbBNVarce.data(), dsz * sizeof(SN_API::snFloat));
                if (oprExist)
                   bn.varce = dbBNVarce.data();
            }
            else if (opr[2] == "scale"){
                dbBNScale.resize(dsz);
                ifs.read((char*)dbBNScale.data(), dsz * sizeof(SN_API::snFloat));
                if (oprExist)
                   bn.scale = dbBNScale.data();
            }
            else if (opr[2] == "schift"){
                dbBNShift.resize(dsz);
                ifs.read((char*)dbBNShift.data(), dsz * sizeof(SN_API::snFloat));
                if (oprExist)
                   bn.schift = dbBNShift.data();
            }
        }

        if (bn.mean &&  bn.varce &&  bn.scale &&  bn.schift){

            operats_[nm]->setBatchNorm(bn);

            bn = batchNorm();
        }

    }

    return true;
}