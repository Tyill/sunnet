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
    Fully connected layer
    */
    class FullyConnected{

    public:
        
        FullyConnected(uint32_t kernel,                           ///< Number of hidden neurons. !Required parameter [0..)
                       calcMode mode = calcMode::CPU,             ///< Сalculation mode. Optional parameter           
                       active act = active::relu,                 ///< Activation function type. Optional parameter
                       weightInit wini = weightInit::he,          ///< Type of initialization of weights. Optional parameter
                       batchNormType bnorm = batchNormType::none, ///< Type of batch norm. Optional parameter
                       snFloat batchNormLr = 0.001,               ///< Learning rate for batch norm coef. Optional parameter [0..)
                       optimizer opt = optimizer::adam,           ///< Optimizer of weights. Optional parameter
                       snFloat decayMomentDW = 0.9,               ///< Optimizer of weights moment change. Optional parameter [0..1.F]
                       snFloat decayMomentWGr = 0.99,             ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
                       snFloat lmbRegular = 0.001,                ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
                       snFloat dropOut = 0.0,                     ///< Random disconnection of neurons. Optional parameter [0..1.F]
                       bool freeze = false,                       ///< Do not change weights. Optional parameter
                       bool gpuClearMem = false)                  ///< Clear memory GPU. Optional parameter
        {
            std::stringstream ss;
            ss << "{\"kernel\":\"" << kernel << "\","
                "\"active\":\"" << activeStr(act) << "\","
                "\"weightInit\"" << weightInitStr(wini) << "\","
                "\"batchNorm\"" << batchNormTypeStr(bnorm) << "\","
                "\"batchNormLr\":\"" << batchNormLr << "\","
                "\"optimizer\":\"" << optimizerStr(opt) << "\","
                "\"decayMomentDW\":\"" << decayMomentDW << "\","
                "\"decayMomentWGr\":\"" << decayMomentWGr << "\","
                "\"lmbRegular\":\"" << lmbRegular << "\","
                "\"dropOut\":\"" << dropOut << "\","
                "\"mode\":\"" << calcModeStr(mode) << "\","
                "\"freeze\":\"" << (freeze ? 1 : 0) << "\","
                "\"gpuClearMem\":\"" << (gpuClearMem ? 1 : 0) << "\""
                "}";

            prm_ = ss.str();
        };

        ~FullyConnected(){};
              
        std::string getParamsJn(){

            return prm_;
        }

        std::string name(){
            return "FullyConnected";
        }

    private:       
        std::string prm_;
    };

    /*
    Convolution layer
    */
    class Convolution{

    public:
        
        Convolution(uint32_t kernel,                   ///< Number of output layers. !Required parameter [0..)
            calcMode mode = calcMode::CPU,             ///< Сalculation mode. Optional parameter
            uint32_t fWidth = 3,                       ///< Width of mask. Optional parameter(> 0)
            uint32_t fHeight = 3,                      ///< Height of mask. Optional parameter(> 0)
            uint32_t padding = 0,                      ///< Padding around the edges. Optional parameter
            uint32_t stride = 1,                       ///< Mask movement step. Optional parameter(> 0)
            uint32_t dilate = 1,                       ///< Expansion mask (> 0). Optional parameter(> 0)
            active act = active::relu,                 ///< Activation function type. Optional parameter
            weightInit wini = weightInit::he,          ///< Type of initialization of weights. Optional parameter
            batchNormType bnorm = batchNormType::none, ///< Type of batch norm. Optional parameter
            snFloat batchNormLr = 0.001,               ///< Learning rate for batch norm coef. Optional parameter [0..)
            optimizer opt = optimizer::adam,           ///< Optimizer of weights. Optional parameter
            snFloat decayMomentDW = 0.9,               ///< Optimizer of weights moment change. Optional parameter [0..1.F]
            snFloat decayMomentWGr = 0.99,             ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
            snFloat lmbRegular = 0.001,                ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
            snFloat dropOut = 0.0,                     ///< Random disconnection of neurons. Optional parameter [0..1.F]
            bool freeze = false,                       ///< Do not change weights. Optional parameter
            bool gpuClearMem = false){                 ///< Clear memory GPU. Optional parameter
                    
            std::stringstream ss;
            ss << "{\"kernel\":\"" << kernel << "\","
                "\"fWidth\":\"" << fWidth << "\","
                "\"fHeight\":\"" << fHeight << "\","
                "\"padding\":\"" << padding << "\","
                "\"stride\":\"" << stride << "\","
                "\"dilate\":\"" << dilate << "\","
                "\"active\":\"" << activeStr(act) << "\","
                "\"weightInit\"" << weightInitStr(wini) << "\","
                "\"batchNorm\"" << batchNormTypeStr(bnorm) << "\","
                "\"batchNormLr\":\"" << batchNormLr << "\","
                "\"optimizer\":\"" << optimizerStr(opt) << "\","
                "\"decayMomentDW\":\"" << decayMomentDW << "\","
                "\"decayMomentWGr\":\"" << decayMomentWGr << "\","
                "\"lmbRegular\":\"" << lmbRegular << "\","
                "\"dropOut\":\"" << dropOut << "\","
                "\"mode\":\"" << calcModeStr(mode) << "\","
                "\"freeze\":\"" << (freeze ? 1 : 0) << "\","
                "\"gpuClearMem\":\"" << (gpuClearMem ? 1 : 0) << "\""
                "}";

            prm_ = ss.str();
        };

        ~Convolution(){};            
      
        std::string getParamsJn(){

            return prm_;
        }

        std::string name(){
            return "Convolution";
        }

    private:        
        std::string prm_;
    };

    /*
    Deconvolution layer
    */
    class Deconvolution{

    public:

        Deconvolution(uint32_t kernel,                 ///< Number of output layers. !Required parameter [0..)
            calcMode mode = calcMode::CPU,             ///< Сalculation mode. Optional parameter
            uint32_t fWidth = 3,                       ///< Width of mask. Optional parameter(> 0)
            uint32_t fHeight = 3,                      ///< Height of mask. Optional parameter(> 0)
            uint32_t stride = 1,                       ///< Mask movement step. Optional parameter(> 0)
            active act = active::relu,                 ///< Activation function type. Optional parameter
            weightInit wini = weightInit::he,          ///< Type of initialization of weights. Optional parameter
            batchNormType bnorm = batchNormType::none, ///< Type of batch norm. Optional parameter
            snFloat batchNormLr = 0.001,               ///< Learning rate for batch norm coef. Optional parameter [0..)
            optimizer opt = optimizer::adam,           ///< Optimizer of weights. Optional parameter
            snFloat decayMomentDW = 0.9,               ///< Optimizer of weights moment change. Optional parameter [0..1.F]
            snFloat decayMomentWGr = 0.99,             ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
            snFloat lmbRegular = 0.001,                ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
            snFloat dropOut = 0.0,                     ///< Random disconnection of neurons. Optional parameter [0..1.F]
            bool freeze = false,                       ///< Do not change weights. Optional parameter
            bool gpuClearMem = false){                 ///< Clear memory GPU. Optional parameter

            std::stringstream ss;
            ss << "{\"kernel\":\"" << kernel << "\","
                "\"fWidth\":\"" << fWidth << "\","
                "\"fHeight\":\"" << fHeight << "\","
                "\"stride\":\"" << stride << "\","
                "\"active\":\"" << activeStr(act) << "\","
                "\"weightInit\"" << weightInitStr(wini) << "\","
                "\"batchNorm\"" << batchNormTypeStr(bnorm) << "\","
                "\"batchNormLr\":\"" << batchNormLr << "\","
                "\"optimizer\":\"" << optimizerStr(opt) << "\","
                "\"decayMomentDW\":\"" << decayMomentDW << "\","
                "\"decayMomentWGr\":\"" << decayMomentWGr << "\","
                "\"lmbRegular\":\"" << lmbRegular << "\","
                "\"dropOut\":\"" << dropOut << "\","
                "\"mode\":\"" << calcModeStr(mode) << "\","
                "\"freeze\":\"" << (freeze ? 1 : 0) << "\","
                "\"gpuClearMem\":\"" << (gpuClearMem ? 1 : 0) << "\""
                "}";

            prm_ = ss.str();
            };

        ~Deconvolution(){};
  
        std::string getParamsJn(){

            return prm_;
        }

        std::string name(){
            return "Deconvolution";
        }

    private:       
        std::string prm_;
    };

    /*
    Pooling layer
    */
    class Pooling{

    public:
               
        Pooling(uint32_t kernel = 2,          ///< Square Mask Size. Optional parameter (> 0) 
            calcMode mode = calcMode::CPU,    ///< Сalculation mode. Optional parameter
            poolType pool = poolType::max,    ///< Operator Type. Optional parameter 
            bool gpuClearMem = false){        ///< Clear memory GPU. Optional parameter

            std::stringstream ss;
            ss << "{\"kernel\":\"" << kernel << "\","
                "\"pool\":\"" << poolTypeStr(pool) << "\","
                "\"mode\":\"" << calcModeStr(mode) << "\","
                "\"gpuClearMem\":\"" << (gpuClearMem ? 1 : 0) << "\""
                "}";

            prm_ = ss.str();
        }

        ~Pooling(){};
                
        std::string getParamsJn(){

            return prm_;
        }

        std::string name(){
            return "Pooling";
        }

    private:      
        std::string prm_;
    };

    /*
    Operator to block further calculation at the current location.
    It is designed for the ability to dynamically disconnect the parallel
    branches of the network during operation.
    */
    class Lock{

    public:
               
        Lock(lockType lock){     ///< Blocking activity. Optional parameter
        
            std::stringstream ss;
            ss << "{\"state\":\"" << lockTypeStr(lock) << "\""
                "}";

            prm_ = ss.str();
        };

        ~Lock(){};
               
        std::string getParamsJn(){
                      
            return prm_;
        }

        std::string name(){
            return "Lock";
        }

    private:        
        std::string prm_;
    };

    /*
    Operator for transferring data to several nodes at once.
    Data can only be received from one node.
    */
    class Switch{

    public:

       
        Switch(std::string nextWay = ""){
        
            std::stringstream ss;
            ss << "{\"nextWay\":\"" << nextWay << "\""
                "}";

            prm_ = ss.str();
        };

        ~Switch(){};
        
        std::string getParamsJn(){

            return prm_;
        }

        std::string name(){
            return "Switch";
        }

    private:
        std::string prm_;
    };

    /*
    The operator is designed to combine the values of two layers.
    The consolidation can be performed by the following options: "summ", "diff", "mean".
    The dimensions of the input layers must be the same.
    */
    class Summator{

    public:
         
        Summator(summatorType summType = summatorType::summ){
          
            std::stringstream ss;
            ss << "{\"type\":\"" << summatorTypeStr(summType) << "\""
                "}";

            prm_ = ss.str();
        };

        ~Summator(){};
             
        std::string getParamsJn(){

            return prm_;
        }

        std::string name(){
            return "Summator";
        }

    private:      
        std::string prm_;
    };

    /*
   
    */
    class Concat{

    public:

        struct params{
          
            params(){
                
            }
        };

        Concat(const params& prm) : prm_(prm){};

        ~Concat(){};
              
        std::string getParamsJn(){

            std::stringstream ss;
         //   ss << "{\"type\":\"" << summatorTypeStr(prm_.summType) << "\""
         //       "}";

            return ss.str();
        }

        std::string name(){
            return "Concat";
        }

    private:        
        params prm_;
    };

    /*

    */
    class Crop{

    public:

        struct rect{
            uint32_t x, y, w, h;
        };        

        Crop(rect rct){
          
            std::stringstream ss;
            ss << "{\"roi\":\"" << rct.x << " " << rct.y << " " << rct.w << " " << rct.h << "\""
                "}";

            prm_ = ss.str();
        };

        ~Crop(){};
               
        std::string getParamsJn(){

            return prm_;
        }
        
        std::string name(){
            return "Crop";
        }

    private:       
        std::string prm_;
    };

    /*

    */
    class UserLayer{

    public:
       
        UserLayer(std::string cbackName){
            
            std::stringstream ss;
            ss << "{\"cbackName\":\"" << cbackName << "\""
                "}";

            prm_ = ss.str();
        };

        ~UserLayer(){};
              
        std::string getParamsJn(){

            return prm_;
        }

        std::string name(){
            return "UserLayer";
        }

    private:     
        std::string prm_;
    };

}
