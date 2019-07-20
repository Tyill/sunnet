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
#include <iostream>
#include <map>

#define ASSERT_MESS(condition, message)                                   \
      if (!(condition)) {                                                 \
         std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
         << " line " << __LINE__ << ": " << (message) << std::endl;       \
         std::abort();                                                    \
      }

namespace SN_Base{
    
    typedef float snFloat;
    
    /// network mode - forward / reverse
    enum class snAction{
        forward = 0,
        backward = 1,
    };
            
    /// size
    struct snSize{
        size_t w, h, d, n, p;
                
        snSize(size_t w_ = 1, size_t h_ = 1, size_t d_ = 1, size_t n_ = 1, size_t p_ = 1) :
            w(w_), h(h_), d(d_), n(n_), p(p_){}

        size_t size() const{
            return w * h * d * n * p;
        }
                
        friend bool operator==(const snSize& left, const snSize& right){

            return (left.w == right.w) && (left.h == right.h) && (left.d == right.d) 
                && (left.n == right.n) && (left.p == right.p);
        }

        friend bool operator!=(const snSize& left, const snSize& right){

            return (left.w != right.w) || (left.h != right.h) || (left.d != right.d)
                || (left.n != right.n) || (left.p != right.p);
        }

    };
        
    /// tensor - input data and output data of each node of the network.
    /// snOperatorCPU/tensor.cpp - implementation CPU
    /// snOperatorCUDA/tensor.cu - implementation CUDA
    struct Tensor{
        
        explicit Tensor(const snSize& sz = snSize(0, 0, 0, 0, 0));

        ~Tensor();

        Tensor(const Tensor& other);
        
        friend bool operator==(const Tensor& left, const Tensor& right){
                        
            return left.sz_ == right.sz_;
        }

        friend bool operator!=(const Tensor& left, const Tensor& right){

            return left.sz_ != right.sz_;
        }

        Tensor& operator=(const Tensor& other);
        
        Tensor& operator+=(const Tensor& other);

        Tensor& operator-=(const Tensor& other);
        
        void setDataGPU(const snFloat* data, const snSize& nsz);

        snFloat* getDataGPU() const;
        
        void setDataCPU(const snFloat* data, const snSize& nsz);

        snFloat* getDataCPU() const;

        void resize(const snSize& nsz);
                
        snSize size() const{

            return sz_;
        }
        
    private:
        mutable snFloat* dataCPU_ = nullptr,
                       * dataGPU_ = nullptr;

        snSize sz_;
    };

    /// parameters of the current operation
    struct operationParam{

        bool isLerning;       ///< lerning
        snAction action;      ///< mode
        snFloat lr;           ///< learning rate
        
        operationParam(bool isLerning_ = false, snAction action_ = snAction::forward, SN_Base::snFloat lr_ = 0.001) :
           isLerning(isLerning_), action(action_), lr(lr_){}
    };
    
    /// layer normalization
    struct batchNorm{
               
       SN_Base::snFloat* norm = nullptr;       ///< norm
       SN_Base::snFloat* mean = nullptr;       ///< mean
       SN_Base::snFloat* varce = nullptr;      ///< disp
       SN_Base::snFloat* scale = nullptr;      ///< γ
       SN_Base::snFloat* dScale = nullptr;     ///< dγ
       SN_Base::snFloat* schift = nullptr;     ///< β
       SN_Base::snFloat* dSchift = nullptr;    ///< dβ
       SN_Base::snFloat* onc = nullptr;        ///< 1 vector 
       SN_Base::snFloat lr = 0.001F;           ///< lrate for γ и β
       snSize sz = snSize(0,0,0,0,0);
             
       void offset(size_t offs){
           mean += offs;
           varce += offs;
           scale += offs;
           dScale += offs;
           schift += offs;
           dSchift += offs;
       }
    };

    /// basic network operator. All settlement operators are inherited from it.
    class OperatorBase{

    protected:
        /// Operator 
        /// @param name - The name of the operator is a specific implementation class
        /// @param node - the name of the node in the NN structure
        /// @param prms - params. Key - name param
        OperatorBase(void* Net_, const std::string& name, const std::string& node, std::map<std::string, std::string>& prms) :
            Net(Net_), name_(name), node_(node), basePrms_(prms){}
        virtual ~OperatorBase(){} 
    public:
        
        /// back link to the parent network object
        void* Net = nullptr;

        virtual bool setInternPrm(std::map<std::string, std::string>& prms){
            basePrms_ = prms;
            return true;
        }

        virtual bool setInput(const snFloat* data, const snSize& dsz){
            baseInput_.setDataCPU(data, dsz);
            return true;
        }

        virtual bool setGradient(const snFloat* data, const snSize& dsz){
            baseGrad_.setDataCPU(data, dsz);
            return true;
        }
        
        virtual bool setWeight(const snFloat* data, const snSize& dsz){
            baseWeight_.setDataCPU(data, dsz);
            return true;
        }

        virtual bool setBatchNorm(const batchNorm& bn){
            baseBatchNorm_ = bn;
            return true;
        }

        virtual std::map<std::string, std::string> getInternPrm() const final{
            return basePrms_;
        }

        virtual const SN_Base::Tensor& getWeight() const final{
            return baseWeight_;
        }

        virtual batchNorm getBatchNorm() const{
            return baseBatchNorm_;
        }

        virtual const SN_Base::Tensor& getOutput() const final{
            return baseOut_;
        }

        virtual const SN_Base::Tensor& getGradient() const final{
            return baseGrad_;
        }

        /// node name in character structure
        virtual std::string node() const final{
            return node_;
        }

        /// The name of the operator is a specific implementation class
        virtual std::string name() const final{
            return name_;
        }

        /// calculation
        /// @param learnPrm - learning options on the iteration
        /// @param neighbOpr - The neighboring operators that transmit data here
        /// @return - list track nodes where to go, if the trace is> 1. If nothing is selected go to all
        virtual std::vector<std::string> Do(const operationParam& learnPrm, const std::vector<OperatorBase*>& neighbOpr) = 0;
        
    protected:
        std::string node_;                            ///< The name of the node in which the statement is evaluated
        std::string name_;                            ///< The name of the operator is a specific implementation class
        std::map<std::string, std::string> basePrms_; ///< param's
        SN_Base::Tensor baseInput_ ;                  ///< input
        SN_Base::Tensor baseWeight_;                  ///< weight
        SN_Base::Tensor baseGrad_ ;                   ///< gradient
        SN_Base::Tensor baseOut_ ;                    ///< output
        batchNorm baseBatchNorm_;                     ///< batch norm
    };

    /// node in the symbol structure of the NN
    struct Node{
        std::string name;                             ///< the name of the node is to be unique within the branch, without the '' and '-'. "Begin", "End" are reserved as the beginning, the end of the network
        std::string oprName;                          ///< the node operator that is executed at the node. One operator per node.
        std::map<std::string, std::string> oprPrms;   ///< parameters of the operator (specifies the user when creating the net)
        std::vector<std::string> prevNodes;           ///< previous nodes (sets of numbers, mk node being collective of non-branches)
        std::vector<std::string> nextNodes;           ///< all the possible trace nodes (set number, mk can be split into several parallel threads), the name of the node trail (via a space) returns the node's operator-node on the iteration. If nothing is returned, it goes to all
    };
    
    /// character structure of NN
    struct Net{            
        std::map<std::string, Node> nodes;            ///< the general collection of nodes of the NN. Key - the name of the node
        std::map<std::string, OperatorBase*> operats; ///< general collection of NN operators. Key - the name of the node
    };

    
    
};
