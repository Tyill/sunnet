//
// sunnet project
// Copyright (C) 2018 by Contributors <https://github.com/Tyill/sunnet>
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
#include "../src/sunnet/sunnet.h"

namespace SN_API{
       
    /*
    Input layer.
    */
    class Input{

    public:

        Input(){ };

        ~Input(){};

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{"
                "}";

            return  ss.str();
        }

        std::string name(){
            return "Input";
        }
    };

    /*
    Fully connected layer
    */
    class FullyConnected{

    public:

        uint32_t units;                            ///< Number of out neurons. !Required parameter [0..)
        active act = active::relu;                 ///< Activation function type. Optional parameter
        optimizer opt = optimizer::adam;           ///< Optimizer of weights. Optional parameter
        snFloat dropOut = 0.0;                     ///< Random disconnection of neurons. Optional parameter [0..1.F]
        batchNormType bnorm = batchNormType::none; ///< Type of batch norm. Optional parameter
        uint32_t gpuDeviceId = 0;                  ///< GPU Id. Optional parameter
        bool freeze = false;                       ///< Do not change weights. Optional parameter
        bool useBias = true;                       ///< +bias. Optional parameter
        weightInit wini = weightInit::he;          ///< Type of initialization of weights. Optional parameter
        snFloat decayMomentDW = 0.9F;              ///< Optimizer of weights moment change. Optional parameter [0..1.F]
        snFloat decayMomentWGr = 0.99F;            ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
        snFloat lmbRegular = 0.001F;               ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
        snFloat batchNormLr = 0.001F;              ///< Learning rate for batch norm coef. Optional parameter [0..)
        
        FullyConnected(uint32_t units_,
                       active act_ = active::relu,                
                       optimizer opt_ = optimizer::adam,          
                       snFloat dropOut_ = 0.0,                    
                       batchNormType bnorm_ = batchNormType::none,
                       uint32_t gpuDeviceId_ = 0):
                       
            units(units_), act(act_), opt(opt_),
            dropOut(dropOut_), bnorm(bnorm_), gpuDeviceId(gpuDeviceId_){};

        FullyConnected(uint32_t units_, batchNormType bnorm_) :
            units(units_), bnorm(bnorm_){}

        ~FullyConnected(){};
              
        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"units\":\"" << units << "\","
                "\"active\":\"" << activeStr(act) << "\","
                "\"weightInit\":\"" << weightInitStr(wini) << "\","
                "\"batchNorm\":\"" << batchNormTypeStr(bnorm) << "\","
                "\"batchNormLr\":\"" << batchNormLr << "\","
                "\"optimizer\":\"" << optimizerStr(opt) << "\","
                "\"decayMomentDW\":\"" << decayMomentDW << "\","
                "\"decayMomentWGr\":\"" << decayMomentWGr << "\","
                "\"lmbRegular\":\"" << lmbRegular << "\","
                "\"dropOut\":\"" << dropOut << "\","                
                "\"gpuDeviceId\":\"" << gpuDeviceId << "\","
                "\"freeze\":\"" << (freeze ? 1 : 0) << "\","
                "\"useBias\":\"" << (useBias ? 1 : 0) << "\""                
                "}";

           return ss.str();
        }

        std::string name(){
            return "FullyConnected";
        }   
    };

    /*
    Convolution layer
    */
    class Convolution{

    public:
        
        uint32_t filters;                          ///< Number of output layers. !Required parameter [0..)
        active act = active::relu;                 ///< Activation function type. Optional parameter
        optimizer opt = optimizer::adam;           ///< Optimizer of weights. Optional parameter
        snFloat dropOut = 0.0;                     ///< Random disconnection of neurons. Optional parameter [0..1.F]
        batchNormType bnorm = batchNormType::none; ///< Type of batch norm. Optional parameter
        uint32_t fWidth = 3;                       ///< Width of mask. Optional parameter(> 0)
        uint32_t fHeight = 3;                      ///< Height of mask. Optional parameter(> 0)
        int padding = 0;                           ///< Padding around the edges. Optional parameter
        uint32_t stride = 1;                       ///< Mask movement step. Optional parameter(> 0)
        uint32_t dilate = 1;                       ///< Expansion mask. Optional parameter(> 0)
        uint32_t gpuDeviceId = 0;                  ///< GPU Id. Optional parameter
        bool freeze = false;                       ///< Do not change weights. Optional parameter
        bool useBias = true;                       ///< +bias. Optional parameter
        weightInit wini = weightInit::he;          ///< Type of initialization of weights. Optional parameter
        snFloat decayMomentDW = 0.9F;              ///< Optimizer of weights moment change. Optional parameter [0..1.F]
        snFloat decayMomentWGr = 0.99F;            ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
        snFloat lmbRegular = 0.001F;               ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
        snFloat batchNormLr = 0.001F;              ///< Learning rate for batch norm coef. Optional parameter [0..)


        Convolution(uint32_t filters_,              
            active act_ = active::relu,                
            optimizer opt_ = optimizer::adam,          
            snFloat dropOut_ = 0.0,                    
            batchNormType bnorm_ = batchNormType::none,
            uint32_t fWidth_ = 3,                      
            uint32_t fHeight_ = 3,                    
            int padding_ = 0,
            uint32_t stride_ = 1,                      
            uint32_t dilate_ = 1,
            uint32_t gpuDeviceId_ = 0):
            
            filters(filters_), act(act_), opt(opt_), dropOut(dropOut_), bnorm(bnorm_),
            fWidth(fWidth_), fHeight(fHeight_), padding(padding_), stride(stride_),
            dilate(dilate_), gpuDeviceId(gpuDeviceId_){}           
       
        Convolution(uint32_t filters_, uint32_t kernelSz, int padding_ = 0, uint32_t stride_ = 1,
            batchNormType bnorm_ = batchNormType::none, active act_ = active::relu) :
            filters(filters_), fWidth(kernelSz), fHeight(kernelSz), padding(padding_), stride(stride_),
            bnorm(bnorm_), act(act_){}

        Convolution(uint32_t filters_, int padding_ = 0) :
            filters(filters_), padding(padding_){}

        ~Convolution(){};            
      
        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"filters\":\"" << filters << "\","
                "\"fWidth\":\"" << fWidth << "\","
                "\"fHeight\":\"" << fHeight << "\","
                "\"padding\":\"" << padding << "\","
                "\"stride\":\"" << stride << "\","
                "\"dilate\":\"" << dilate << "\","
                "\"active\":\"" << activeStr(act) << "\","
                "\"weightInit\":\"" << weightInitStr(wini) << "\","
                "\"batchNorm\":\"" << batchNormTypeStr(bnorm) << "\","
                "\"batchNormLr\":\"" << batchNormLr << "\","
                "\"optimizer\":\"" << optimizerStr(opt) << "\","
                "\"decayMomentDW\":\"" << decayMomentDW << "\","
                "\"decayMomentWGr\":\"" << decayMomentWGr << "\","
                "\"lmbRegular\":\"" << lmbRegular << "\","
                "\"dropOut\":\"" << dropOut << "\","                
                "\"gpuDeviceId\":\"" << gpuDeviceId << "\","
                "\"freeze\":\"" << (freeze ? 1 : 0) << "\","
                "\"useBias\":\"" << (useBias ? 1 : 0) << "\""                
                "}";

            return ss.str();
        }

        std::string name(){
            return "Convolution";
        }
          
    };

    /*
    Deconvolution layer
    */
    class Deconvolution{

    public:

        uint32_t filters;                          ///< Number of output layers. !Required parameter [0..)
        active act = active::relu;                 ///< Activation function type. Optional parameter
        optimizer opt = optimizer::adam;           ///< Optimizer of weights. Optional parameter
        snFloat dropOut = 0.0;                     ///< Random disconnection of neurons. Optional parameter [0..1.F]
        batchNormType bnorm = batchNormType::none; ///< Type of batch norm. Optional parameter
        uint32_t fWidth = 3;                       ///< Width of mask. Optional parameter(> 0)
        uint32_t fHeight = 3;                      ///< Height of mask. Optional parameter(> 0)
        uint32_t stride = 2;                       ///< Mask movement step. Optional parameter(> 0)                
        uint32_t gpuDeviceId = 0;                  ///< GPU Id. Optional parameter      
        bool freeze = false;                       ///< Do not change weights. Optional parameter
        weightInit wini = weightInit::he;          ///< Type of initialization of weights. Optional parameter
        snFloat decayMomentDW = 0.9F;              ///< Optimizer of weights moment change. Optional parameter [0..1.F]
        snFloat decayMomentWGr = 0.99F;            ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
        snFloat lmbRegular = 0.001F;               ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
        snFloat batchNormLr = 0.001F;              ///< Learning rate for batch norm coef. Optional parameter [0..)
        
        Deconvolution(uint32_t filters_,
            active act_ = active::relu,
            optimizer opt_ = optimizer::adam,
            snFloat dropOut_ = 0.0,
            batchNormType bnorm_ = batchNormType::none,
            uint32_t fWidth_ = 3,
            uint32_t fHeight_ = 3,
            uint32_t stride_ = 2,           
            uint32_t gpuDeviceId_ = 0):
            
            filters(filters_), act(act_), opt(opt_), dropOut(dropOut_), bnorm(bnorm_),
            fWidth(fWidth_), fHeight(fHeight_), stride(stride_),
            gpuDeviceId(gpuDeviceId_){}
       
        Deconvolution(uint32_t filters_) :
            filters(filters_){}

        ~Deconvolution(){};
  
        std::string getParamsJn(){
            
            std::stringstream ss;
            ss << "{\"filters\":\"" << filters << "\","
                "\"fWidth\":\"" << fWidth << "\","
                "\"fHeight\":\"" << fHeight << "\","
                "\"stride\":\"" << stride << "\","
                "\"active\":\"" << activeStr(act) << "\","
                "\"weightInit\":\"" << weightInitStr(wini) << "\","
                "\"batchNorm\":\"" << batchNormTypeStr(bnorm) << "\","
                "\"batchNormLr\":\"" << batchNormLr << "\","
                "\"optimizer\":\"" << optimizerStr(opt) << "\","
                "\"decayMomentDW\":\"" << decayMomentDW << "\","
                "\"decayMomentWGr\":\"" << decayMomentWGr << "\","
                "\"lmbRegular\":\"" << lmbRegular << "\","
                "\"dropOut\":\"" << dropOut << "\","                
                "\"gpuDeviceId\":\"" << gpuDeviceId << "\","
                "\"freeze\":\"" << (freeze ? 1 : 0) << "\""               
                "}";

           return ss.str();
        }

        std::string name(){
            return "Deconvolution";
        }

    };

    /*
    Pooling layer
    */
    class Pooling{

    public:
            
        uint32_t kernel = 2;              ///< Square Mask Size. Optional parameter (> 0) 
        uint32_t stride = 2;              ///< Mask movement step. Optional parameter(> 0)
        poolType pool = poolType::max;    ///< Operator Type. Optional parameter 
        uint32_t gpuDeviceId = 0;         ///< GPU Id. Optional parameter
   
        Pooling(uint32_t gpuDeviceId_ = 0):            
            gpuDeviceId(gpuDeviceId_){}
              
        Pooling(uint32_t kernel_, uint32_t stride_, poolType pool_ = poolType::max) :
            kernel(kernel_), stride(stride_), pool(pool_){}

        ~Pooling(){};
                
        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"kernel\":\"" << kernel << "\","
                "\"stride\":\"" << stride << "\","
                "\"pool\":\"" << poolTypeStr(pool) << "\","
                "\"gpuDeviceId\":\"" << gpuDeviceId << "\""                
                "}";

            return ss.str();
        }

        std::string name(){
            return "Pooling";
        }
    };

    /*
    Operator to block further calculation at the current location.
    It is designed for the ability to dynamically disconnect the parallel
    branches of the network during operation.
    */
    class Lock{

    public:
             
        lockType lockTp;    ///< Blocking activity. Optional parameter

        Lock(lockType lockTp_) : lockTp(lockTp_){}
        
        ~Lock(){};
               
        std::string getParamsJn(){
                      
            std::stringstream ss;
            ss << "{\"state\":\"" << lockTypeStr(lockTp) << "\""
                "}";

            return ss.str();
        }

        std::string name(){
            return "Lock";
        }
    };

    /*
    Operator for transferring data to several nodes at once.
    Data can only be received from one node.
    */
    class Switch{

    public:

        std::string nextWay;   // next nodes through a space
       
        Switch(const std::string& nextWay_) :nextWay(nextWay_){};

        ~Switch(){};
        
        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"nextWay\":\"" << nextWay << "\""
                "}";

            return ss.str();
        }

        std::string name(){
            return "Switch";
        }
    };

    /*
    The operator is designed to combine the values of two layers.
    The consolidation can be performed by the following options: "summ", "diff", "mean".
    The dimensions of the input layers must be the same.
    */
    class Summator{

    public:
         
        summatorType summType;     

        Summator(summatorType summType_ = summatorType::summ) : summType(summType_){};

        ~Summator(){};
             
        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"type\":\"" << summatorTypeStr(summType) << "\""
                "}";

            return ss.str();
        }

        std::string name(){
            return "Summator";
        }
    };

    /*
    The operator connects the channels with multiple layers
    */
    class Concat{

    public:
              
        std::string sequence;    // prev nodes through a space

        Concat(const std::string& sequence_) : sequence(sequence_){};

        ~Concat(){};
              
        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"sequence\":\"" << sequence << "\""
                "}";

            return ss.str();
        }

        std::string name(){
            return "Concat";
        }
    };

    /*
    Change the number of channels
    */
    class Resize{

    public:
               
        diap fwdDiap, bwdDiap;   // diap layer through a space

        Resize(const diap& fwdDiap_, const diap& bwdDiap_) :
            fwdDiap(fwdDiap_), bwdDiap(bwdDiap_){};

        ~Resize(){};

        std::string getParamsJn(){
                       
            std::stringstream ss;
            ss << "{\"fwdDiap\":\"" << fwdDiap.begin << " " << fwdDiap.end << "\","
                "\"bwdDiap\":\"" << bwdDiap.begin << " " << bwdDiap.end << "\""
                "}";

            return ss.str();
        }

        std::string name(){
            return "Resize";
        }
    };

    /*
    ROI clipping in each image of each channel
    */
    class Crop{

    public:
        
        rect rct;         // region of interest

        Crop(const rect& rct_) : rct(rct_){};

        ~Crop(){};
               
        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"roi\":\"" << rct.x << " " << rct.y << " " << rct.w << " " << rct.h << "\""
                "}";

            return ss.str();
        }
        
        std::string name(){
            return "Crop";
        }
    };

    /*
    Activation function operator
    */
    class Activation{

    public:

        active act = active::relu;                 ///< Activation function type. Optional parameter

        Activation(const active& act_) : act(act_){};

        ~Activation(){};

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"active\":\"" << activeStr(act) + "\"}";

            return ss.str();
        }

        std::string name(){
            return "Activation";
        }
    };

    /*
    Custom layer
    */
    class UserLayer{

    public:
       
        std::string cbackName;

        UserLayer(const std::string& cbackName_) : cbackName(cbackName_){};

        ~UserLayer(){};
              
        std::string getParamsJn(){
            
            std::stringstream ss;
            ss << "{\"cbackName\":\"" << cbackName << "\""
                "}";

            return ss.str();
        }

        std::string name(){
            return "UserLayer";
        }
    };

    /*
    Error function calculation layer
    */
    class LossFunction{

    public:
           
        lossType loss;

        LossFunction(lossType loss_) : loss(loss_){};

        ~LossFunction(){};

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{\"loss\":\"" << lossTypeStr(loss) << "\""
                "}";

            return ss.str();
        }

        std::string name(){
            return "LossFunction";
        }
    };

    /*
    Batch norm
    */
    class BatchNormLayer{

    public:
      
        BatchNormLayer(){};

        ~BatchNormLayer(){};

        std::string getParamsJn(){

            std::stringstream ss;
            ss << "{}";

            return ss.str();
        }

        std::string name(){
            return "BatchNorm";
        }
    };

}
