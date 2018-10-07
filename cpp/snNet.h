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


#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include "../src/skynet/skyNet.h"
#include "snTensor.h"

namespace SN_API{
      
    /// version library
    /// @return version
    std::string versionLib(){

        char ver[32];
        snVersionLib(ver);

        return ver;
    }
   
    class Net{
               
    public:

        /// create net
        /// @param[in] jnNet - network architecture in JSON
        /// @param[in] weightPath - path to file with weight
        Net(const std::string& jnNet = "", const std::string& weightPath = ""){
        
            if (!jnNet.empty())
                createNet(jnNet);

            if (net_ && !weightPath.empty())
                loadAllWeightFromFile(weightPath);            
        };

        ~Net(){        
            if (net_)
                snFreeNet(net_);        
        };
               
        /// last error
        /// @return "" ok
        std::string getLastErrorStr(){

            if (net_){
               char err[256]; err[0] = '\0';
               snGetLastErrorStr(net_, err);
               err_ = err;
            }

            return err_;
        }

        /// add node (layer)
        /// @param[in] name - name node in architecture of net
        /// @param[in] nd - tensor node
        /// @param[in] nextNodes - next nodes through a space
        /// @return ref Net
        template<typename T> 
        Net& addNode(const std::string& name, T& nd, const std::string& nextNodes){
                        
            nodes_.push_back(node{ name, nd.name(), nd.getParamsJn(), nextNodes });
            
            return *this;
        }

        /// update param node (layer)
        /// @param[in] name - name node in architecture of net
        /// @param[in] nd - tensor node
        /// @return true - ok
        template<typename T>
        bool updateNode(const std::string& name, const T& nd){

            bool ok = false;
            if (net_)
                ok = snSetParamNode(net_, name.c_str(), nd.getParamsJn());
            else{
                for (auto& nd : nodes_){
                    if (nd.name == name){
                        nd.params = nd.getParamsJn();
                        ok = true;
                        break;
                    }
                }
            }

            return ok;
        }

        /// forward action
        /// @param[in] isLern - is lerning ?
        /// @param[in] inTns - in tensor
        /// @param[inout] outTns - out result tensor
        /// @return true - ok
        bool forward(bool isLern, Tensor& inTns, Tensor& outTns){

            if (!net_ && !createNet()) return false;

            return snForward(net_, isLern, inTns.size(), inTns.data(), outTns.size(), outTns.data());
        }

        /// backward action
        /// @param[in] lr - lerning rate
        /// @param[in] gradTns - grad error tensor
        /// @return true - ok
        bool backward(snFloat lr, Tensor& gradTns){

            if (!net_ && !createNet()) return false;

            return snBackward(net_, lr, gradTns.size(), gradTns.data());
        }

        /// training action - cycle forward-backward
        /// @param[in] lr - lerning rate
        /// @param[in] inTns - in tensor
        /// @param[inout] outTns - out tensor
        /// @param[in] targetTns - target tensor
        /// @param[inout] outAccurate - accurate error
        /// @return true - ok
        bool training(snFloat lr, Tensor& inTns, Tensor& outTns, Tensor& targetTns, snFloat& outAccurate){

            if (!net_ && !createNet()) return false;

            return snTraining(net_, lr, inTns.size(), inTns.data(), 
                outTns.size(), outTns.data(),
                targetTns.data(), &outAccurate);
        }
            
        /// set weight of node
        /// @param[in] name - name node in architecture of net
        /// @param[in] weight - set weight tensor
        /// @return true - ok
        bool setWeightNode(const std::string& name, Tensor& weight){

            if (!net_) return false;

            return snSetWeightNode(net_, name.c_str(), weight.size(), weight.data());
        }

        /// get weight of node
        /// @param[in] name - name node in architecture of net
        /// @param[out] outWeight - weight tensor
        /// @return true - ok
        bool getWeightNode(const std::string& name, Tensor& outWeight){

            if (!net_) return false;

            snLSize wsz; snFloat* wdata = nullptr;
            if (snGetWeightNode(net_, name.c_str(), &wsz, &wdata) && wdata){

                outWeight = Tensor(wsz, wdata);

                snFreeResources(wdata, 0);
                return true;
            }
            else
                return false;
        }
        
        /// save all weight's in file
        /// @param[in] path - file path
        /// @return true - ok
        bool saveAllWeightToFile(const std::string& path){

            if (!net_) return false;

            return snSaveAllWeightToFile(net_, path.c_str());
        }

        /// load all weight's from file
        /// @param[in] path - file path
        /// @return true - ok
        bool loadAllWeightFromFile(const std::string& path){

            if (!net_ && !createNet()) return false;

            return snLoadAllWeightFromFile(net_, path.c_str());
        }

        /// add user callback
        /// @param[in] name - name userCBack in architecture of net
        /// @param[in] cback - call back function
        /// @param[in] udata - aux data
        /// @return true - ok
        bool addUserCBack(const std::string& name, snUserCBack cback, snUData udata){

            bool ok = true;
            if (net_)
               ok = snAddUserCallBack(net_, name.c_str(), cback, udata);
            else
                ucb_.push_back(uCBack{ name, cback, udata });

            return ok;
        }

        /// architecture of net in json
        /// @return jn arch
        std::string getArchitecNetJN(){

            if (!net_ && !createNet()) return "";

            char* arch = nullptr;
            snGetArchitecNet(net_, &arch);
            
            std::string ret = arch;

            snFreeResources(0, arch);

            return ret;
        }

    private:

        std::string err_;

        struct node{
            std::string name;
            std::string opr;
            std::string params;
            std::string nextNodes;
        };

        struct uCBack{
            std::string name;
            snUserCBack cback;
            snUData udata;           
        };

        std::vector<node> nodes_;
        std::vector<uCBack> ucb_;

        skyNet net_ = nullptr;

        std::string netStruct_;

        bool createNet(){

            if (net_) return true;

            if (nodes_.empty()) return false;

            std::string beginNode = nodes_.front().name,
                        prevEndNode = nodes_.back().name;

            for (auto& nd : nodes_){
                if (nd.opr == "Input") beginNode = nd.nextNodes;
                if (nd.nextNodes == "Output"){
                    prevEndNode = nd.name;
                    nd.nextNodes = "EndNet";
                }
            }

            std::stringstream ss;
            ss << "{"
                "\"BeginNet\":"
                "{"
                "\"NextNodes\":\"" + beginNode + "\""
                "},"

                "\"Nodes\":"
                "[";

            size_t sz = nodes_.size();
            for (int i = 0; i < sz; ++i){

                auto& nd = nodes_[i];

                if ((nd.name == "Input") || (nd.name == "Output"))
                    continue;
                                
                ss << "{"
                    "\"NodeName\":\"" + nd.name + "\","
                    "\"NextNodes\":\"" + nd.nextNodes + "\","
                    "\"OperatorName\":\"" + nd.opr + "\","
                    "\"OperatorParams\":" + nd.params + ""
                    "}";

                if (i < sz - 1)  ss << ",";
            }
          
            ss << "],"

                "\"EndNet\":"                         
                "{"
                "\"PrevNode\":\"" + prevEndNode + "\""
                "}"
                "}";
           
           
            return createNet(ss.str().c_str());
        }

        bool createNet(const std::string& jnNet){
            
            if (net_) return true;

            char err[256]; err[0] = '\0';
            net_ = snCreateNet(jnNet.c_str(), err);

            err_ = err;

            if (net_){
                for (auto& cb : ucb_)
                    snAddUserCallBack(net_, cb.name.c_str(), cb.cback, cb.udata);
            }

            return net_ != nullptr;
        }
    };

    
}
