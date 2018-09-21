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
#include <sstream>
#include "snType.h"
#include "../src/skynet/skyNet.h"

namespace SN_API{
       
    /* 
    The input node receives the user data, and transmits further along the chain.
    */
    class Input{

    public:

        Input(skyNet net, const std::string& name) :
            net_(net), name_(name){ }

        ~Input(){};

    private:
        std::string name_;
        skyNet net_ = nullptr;
    };

    /*
    The output node of the network only receives the resulting data.
    */
    class Output{

    public:

        Output(skyNet net, const std::string& name) :
            net_(net), name_(name){ }

        ~Output(){};

    private:
        std::string name_;
        skyNet net_ = nullptr;
    };

    /*
    Fully connected layer
    */
    class FullyConnected{

    public:

        struct params{
            uint32_t kernel = 1;                       ///< Number of hidden neurons. !Required parameter [0..)
            active act = active::relu;                 ///< Activation function type. Optional parameter
            weightInit wini = weightInit::he;          ///< Type of initialization of weights. Optional parameter
            batchNormType bnorm = batchNormType::none; ///< Type of batch norm. Optional parameter
            snFloat batchNormLr = 0.001;               ///< Learning rate for batch norm coef. Optional parameter [0..)
            optimizer opt = optimizer::adam;           ///< Optimizer of weights. Optional parameter
            snFloat decayMomentDW = 0.9;               ///< Optimizer of weights moment change. Optional parameter [0..1.F]
            snFloat decayMomentWGr = 0.99;             ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
            snFloat lmbRegular = 0.001;                ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
            snFloat dropOut = 0.0;                     ///< Random disconnection of neurons. Optional parameter [0..1.F]
            calcMode mode = calcMode::CPU;             ///< Сalculation mode. Optional parameter
            bool freeze = false;                       ///< Do not change weights. Optional parameter
            bool gpuClearMem = false;                  ///< Clear memory GPU. Optional parameter

            params(uint32_t kernel_, calcMode mode_){
                kernel = kernel;
                mode = mode_;
            }
        };

        FullyConnected(skyNet net, const std::string& name, const params& prm) :
            net_(net), name_(name), prm_(prm){ }

        ~FullyConnected(){};

        bool setParams(const params& prm){

            prm_ = prm;
                       
            return snSetParamNode(net_, name_.c_str(), getParamsJn().c_str());
        };

        params getParams(){

            return prm_;
        };

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"kernel\":\""         << prm_.kernel << "\","
                   "\"active\":\""         << activeStr(prm_.act) << "\","
                   "\"weightInit\""        << weightInitStr(prm_.wini) << "\","
                   "\"batchNorm\""         << batchNormTypeStr(prm_.bnorm) << "\","
                   "\"batchNormLr\":\""    << prm_.batchNormLr << "\","
                   "\"optimizer\":\""      << optimizerStr(prm_.opt) << "\","
                   "\"decayMomentDW\":\""  << prm_.decayMomentDW << "\","
                   "\"decayMomentWGr\":\"" << prm_.decayMomentWGr << "\","
                   "\"lmbRegular\":\""     << prm_.lmbRegular << "\","
                   "\"dropOut\":\""        << prm_.dropOut << "\","
                   "\"mode\":\""           << calcModeStr(prm_.mode) << "\","
                   "\"freeze\":\""         << (prm_.freeze ? 1 : 0) << "\","
                   "\"gpuClearMem\":\""    << (prm_.gpuClearMem ? 1 : 0) << "\""
                  "}";

            return ss.str();
        }

    private:
        std::string name_;
        skyNet net_ = nullptr;
        params prm_;
    };

    /*
    Convolution layer
    */
    class Convolution{

    public:

        struct params{
            uint32_t kernel = 1;                       ///< Number of output layers. !Required parameter [0..)
            uint32_t fWidth = 3;                       ///< Width of mask. Optional parameter(> 0)
            uint32_t fHeight = 3;                      ///< Height of mask. Optional parameter(> 0)
            uint32_t padding = 0;                      ///< Padding around the edges. Optional parameter
            uint32_t stride = 1;                       ///< Mask movement step. Optional parameter(> 0)
            uint32_t dilate = 1;                       ///< Expansion mask (> 0). Optional parameter(> 0)
            active act = active::relu;                 ///< Activation function type. Optional parameter
            weightInit wini = weightInit::he;          ///< Type of initialization of weights. Optional parameter
            batchNormType bnorm = batchNormType::none; ///< Type of batch norm. Optional parameter
            snFloat batchNormLr = 0.001;               ///< Learning rate for batch norm coef. Optional parameter [0..)
            optimizer opt = optimizer::adam;           ///< Optimizer of weights. Optional parameter
            snFloat decayMomentDW = 0.9;               ///< Optimizer of weights moment change. Optional parameter [0..1.F]
            snFloat decayMomentWGr = 0.99;             ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
            snFloat lmbRegular = 0.001;                ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
            snFloat dropOut = 0.0;                     ///< Random disconnection of neurons. Optional parameter [0..1.F]
            calcMode mode = calcMode::CPU;             ///< Сalculation mode. Optional parameter
            bool freeze = false;                       ///< Do not change weights. Optional parameter
            bool gpuClearMem = false;                  ///< Clear memory GPU. Optional parameter

            params(uint32_t kernel_, calcMode mode_){
                kernel = kernel;
                mode = mode_;
            }
        };

        Convolution(skyNet net, const std::string& name, const params& prm) :
            net_(net), name_(name), prm_(prm){ }

        ~Convolution(){};

        bool setParams(const params& prm){

            prm_ = prm;

            return snSetParamNode(net_, name_.c_str(), getParamsJn().c_str());
        };

        params getParams(){

            return prm_;
        };

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"kernel\":\"" << prm_.kernel << "\","
                "\"fWidth\":\"" << prm_.fWidth << "\","
                "\"fHeight\":\"" << prm_.fHeight << "\","
                "\"padding\":\"" << prm_.padding << "\","
                "\"stride\":\"" << prm_.stride << "\","
                "\"dilate\":\"" << prm_.dilate << "\","
                "\"active\":\"" << activeStr(prm_.act) << "\","
                "\"weightInit\"" << weightInitStr(prm_.wini) << "\","
                "\"batchNorm\"" << batchNormTypeStr(prm_.bnorm) << "\","
                "\"batchNormLr\":\"" << prm_.batchNormLr << "\","
                "\"optimizer\":\"" << optimizerStr(prm_.opt) << "\","
                "\"decayMomentDW\":\"" << prm_.decayMomentDW << "\","
                "\"decayMomentWGr\":\"" << prm_.decayMomentWGr << "\","
                "\"lmbRegular\":\"" << prm_.lmbRegular << "\","
                "\"dropOut\":\"" << prm_.dropOut << "\","
                "\"mode\":\"" << calcModeStr(prm_.mode) << "\","
                "\"freeze\":\"" << (prm_.freeze ? 1 : 0) << "\","
                "\"gpuClearMem\":\"" << (prm_.gpuClearMem ? 1 : 0) << "\""
                "}";

            return ss.str();
        }

    private:
        std::string name_;
        skyNet net_ = nullptr;
        params prm_;
    };

    /*
    Deconvolution layer
    */
    class Deconvolution{

    public:

        struct params{
            uint32_t kernel = 1;                       ///< Number of output layers. !Required parameter [0..)
            uint32_t fWidth = 3;                       ///< Width of mask. Optional parameter(> 0)
            uint32_t fHeight = 3;                      ///< Height of mask. Optional parameter(> 0)
            uint32_t stride = 1;                       ///< Mask movement step. Optional parameter(> 0)
            active act = active::relu;                 ///< Activation function type. Optional parameter
            weightInit wini = weightInit::he;          ///< Type of initialization of weights. Optional parameter
            batchNormType bnorm = batchNormType::none; ///< Type of batch norm. Optional parameter
            snFloat batchNormLr = 0.001;               ///< Learning rate for batch norm coef. Optional parameter [0..)
            optimizer opt = optimizer::adam;           ///< Optimizer of weights. Optional parameter
            snFloat decayMomentDW = 0.9;               ///< Optimizer of weights moment change. Optional parameter [0..1.F]
            snFloat decayMomentWGr = 0.99;             ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
            snFloat lmbRegular = 0.001;                ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
            snFloat dropOut = 0.0;                     ///< Random disconnection of neurons. Optional parameter [0..1.F]
            calcMode mode = calcMode::CPU;             ///< Сalculation mode. Optional parameter
            bool freeze = false;                       ///< Do not change weights. Optional parameter
            bool gpuClearMem = false;                  ///< Clear memory GPU. Optional parameter

            params(uint32_t kernel_, calcMode mode_){
                kernel = kernel;
                mode = mode_;
            }
        };

        Deconvolution(skyNet net, const std::string& name, const params& prm) :
            net_(net), name_(name), prm_(prm){ }

        ~Deconvolution(){};

        bool setParams(const params& prm){

            prm_ = prm;

            return snSetParamNode(net_, name_.c_str(), getParamsJn().c_str());
        };

        params getParams(){

            return prm_;
        };

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"kernel\":\"" << prm_.kernel << "\","
                "\"fWidth\":\"" << prm_.fWidth << "\","
                "\"fHeight\":\"" << prm_.fHeight << "\","
                "\"stride\":\"" << prm_.stride << "\","
                "\"active\":\"" << activeStr(prm_.act) << "\","
                "\"weightInit\"" << weightInitStr(prm_.wini) << "\","
                "\"batchNorm\"" << batchNormTypeStr(prm_.bnorm) << "\","
                "\"batchNormLr\":\"" << prm_.batchNormLr << "\","
                "\"optimizer\":\"" << optimizerStr(prm_.opt) << "\","
                "\"decayMomentDW\":\"" << prm_.decayMomentDW << "\","
                "\"decayMomentWGr\":\"" << prm_.decayMomentWGr << "\","
                "\"lmbRegular\":\"" << prm_.lmbRegular << "\","
                "\"dropOut\":\"" << prm_.dropOut << "\","
                "\"mode\":\"" << calcModeStr(prm_.mode) << "\","
                "\"freeze\":\"" << (prm_.freeze ? 1 : 0) << "\","
                "\"gpuClearMem\":\"" << (prm_.gpuClearMem ? 1 : 0) << "\""
                "}";

            return ss.str();
        }

    private:
        std::string name_;
        skyNet net_ = nullptr;
        params prm_;
    };

    /*
    Pooling layer
    */
    class Pooling{

    public:

        struct params{
            uint32_t kernel = 1;                       ///< Square Mask Size. Optional parameter (> 0) 
            poolType pool = poolType::max;             ///< Operator Type. Optional parameter 
            calcMode mode = calcMode::CPU;             ///< Сalculation mode. Optional parameter
            bool gpuClearMem = false;                  ///< Clear memory GPU. Optional parameter

            params(uint32_t kernel_, calcMode mode_){
                kernel = kernel;
                mode = mode_;
            }
        };

        Pooling(skyNet net, const std::string& name, const params& prm) :
            net_(net), name_(name), prm_(prm){ }

        ~Pooling(){};

        bool setParams(const params& prm){

            prm_ = prm;

            return snSetParamNode(net_, name_.c_str(), getParamsJn().c_str());
        };

        params getParams(){

            return prm_;
        };

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"kernel\":\"" << prm_.kernel << "\","
                "\"pool\":\"" << poolTypeStr(prm_.pool) << "\","
                "\"mode\":\"" << calcModeStr(prm_.mode) << "\","
                "\"gpuClearMem\":\"" << (prm_.gpuClearMem ? 1 : 0) << "\""
                "}";

            return ss.str();
        }

    private:
        std::string name_;
        skyNet net_ = nullptr;
        params prm_;
    };

    /*
    Operator to block further calculation at the current location.
    It is designed for the ability to dynamically disconnect the parallel
    branches of the network during operation.
    */
    class Look{

    public:

        struct params{
            lockType lock = lockType::unlock;          ///< Blocking activity. Optional parameter
           
            params(lockType lock_ = lockType::unlock){
                lock = lock_;
            }
        };

        Look(skyNet net, const std::string& name, const params& prm) :
            net_(net), name_(name), prm_(prm){ }

        ~Look(){};

        bool setParams(const params& prm){

            prm_ = prm;

            return snSetParamNode(net_, name_.c_str(), getParamsJn().c_str());
        };

        params getParams(){

            return prm_;
        };

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"state\":\"" << lockTypeStr(prm_.lock) << "\""
                  "}";

            return ss.str();
        }

    private:
        std::string name_;
        skyNet net_ = nullptr;
        params prm_;
    };

    /*
    Operator for transferring data to several nodes at once.
    Data can only be received from one node.
    */
    class Switch{

    public:

        struct params{
            std::string nextWay;
            params(std::string nextWay_ = ""){
                nextWay = nextWay_;
            }
        };

        Switch(skyNet net, const std::string& name, const params& prm) :
            net_(net), name_(name), prm_(prm){ }

        ~Switch(){};

        bool setParams(const params& prm){

            prm_ = prm;

            return snSetParamNode(net_, name_.c_str(), getParamsJn().c_str());
        };

        params getParams(){

            return prm_;
        };

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"nextWay\":\"" << prm_.nextWay << "\""
                "}";

            return ss.str();
        }

    private:
        std::string name_;
        skyNet net_ = nullptr;
        params prm_;
    };

    /*
    The operator is designed to combine the values of two layers.
    The consolidation can be performed by the following options: "summ", "diff", "mean".
    The dimensions of the input layers must be the same.
    */
    class Summator{

    public:

        struct params{
            summatorType summType;
            params(summatorType summType_ = summatorType::summ){
                summType = summType_;
            }
        };

        Summator(skyNet net, const std::string& name, const params& prm) :
            net_(net), name_(name), prm_(prm){ }

        ~Summator(){};

        bool setParams(const params& prm){

            prm_ = prm;

            return snSetParamNode(net_, name_.c_str(), getParamsJn().c_str());
        };

        params getParams(){

            return prm_;
        };

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"type\":\"" << summatorTypeStr(prm_.summType) << "\""
                "}";

            return ss.str();
        }

    private:
        std::string name_;
        skyNet net_ = nullptr;
        params prm_;
    };

    /*
   
    */
    class Concat{

    public:

        struct params{
          
            params(){
                
            }
        };

        Concat(skyNet net, const std::string& name, const params& prm) :
            net_(net), name_(name), prm_(prm){ }

        ~Concat(){};

        bool setParams(const params& prm){

            prm_ = prm;

            return snSetParamNode(net_, name_.c_str(), getParamsJn().c_str());
        };

        params getParams(){

            return prm_;
        };

        std::string getParamsJn(){

            std::stringstream ss;
         //   ss << "{\"type\":\"" << summatorTypeStr(prm_.summType) << "\""
         //       "}";

            return ss.str();
        }

    private:
        std::string name_;
        skyNet net_ = nullptr;
        params prm_;
    };

    /*

    */
    class Crop{

    public:

        struct rect{
            uint32_t x, y, w, h;
        };

        struct params{

            rect rct;

            params(rect rct_){
                rct = rct_;
            }
        };

        Crop(skyNet net, const std::string& name, const params& prm) :
            net_(net), name_(name), prm_(prm){ }

        ~Crop(){};

        bool setParams(const params& prm){

            prm_ = prm;

            return snSetParamNode(net_, name_.c_str(), getParamsJn().c_str());
        };

        params getParams(){

            return prm_;
        };

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"roi\":\"" << prm_.rct.x << " " << prm_.rct.y << " " << prm_.rct.w << " " << prm_.rct.h << "\""
                   "}";

            return ss.str();
        }

    private:
        std::string name_;
        skyNet net_ = nullptr;
        params prm_;
    };

    /*

    */
    class UserLayer{

    public:
           
        struct params{

            std::string cbackName;

            params(const std::string& cbackName_){
                cbackName = cbackName_;
            }
        };

        UserLayer(skyNet net, const std::string& name, const params& prm) :
            net_(net), name_(name), prm_(prm){ }

        ~UserLayer(){};

        bool setParams(const params& prm){

            prm_ = prm;

            return snSetParamNode(net_, name_.c_str(), getParamsJn().c_str());
        };

        params getParams(){

            return prm_;
        };

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"cbackName\":\"" << prm_.cbackName << "\""
                "}";

            return ss.str();
        }

    private:
        std::string name_;
        skyNet net_ = nullptr;
        params prm_;
    };

}
