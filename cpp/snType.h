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

namespace SN_API{

    /// Activation function type
    enum class active{
        none = -1,
        sigmoid = 0,
        relu = 1,
        leakyRelu = 2,
        elu = 3,
    };
    std::string activeStr(active act){

        switch (act){
            case active::none:      return "none";
            case active::sigmoid:   return "sigmoid";
            case active::relu:      return "relu";
            case active::leakyRelu: return "leakyRelu";
            case active::elu:       return "elu";
            default:                return "none";
        }
    }

    /// Type of initialization of weights
    enum class weightInit{
        uniform = 0,
        he = 1,
        lecun = 2,
        xavier = 3,
    };
    std::string weightInitStr(weightInit wini){
   
        switch (wini){
            case weightInit::uniform: return "uniform";
            case weightInit::he:      return "he";
            case weightInit::lecun:   return "lecun";
            case weightInit::xavier:  return "xavier";
            default:                  return "uniform";
        }
    }

    /// Type of batch norm 
    enum class batchNormType{
        none = -1,
        beforeActive = 0,
        postActive = 1,
    };
    std::string batchNormTypeStr(batchNormType bnorm){

        switch (bnorm){
            case batchNormType::none:         return "none";
            case batchNormType::beforeActive: return "beforeActive";
            case batchNormType::postActive:   return "postActive";
            default:                          return "none";
        }
    }

    /// Optimizer of weights
    enum class optimizer{
        sgd = 0,
        sgdMoment = 1,
        adagrad = 2,
        RMSprop = 3,
        adam = 4,
    };
    std::string optimizerStr(optimizer opt){

        switch (opt){
        case optimizer::sgd:       return "sgd";
        case optimizer::sgdMoment: return "sgdMoment";
        case optimizer::adagrad:   return "adagrad";
        case optimizer::RMSprop:   return "RMSprop";
        case optimizer::adam:      return "adam";
        default:                   return "adam";
        }
    }
    
    /// pooling
    enum class poolType{
        max = 0,
        avg = 1,
    };
    std::string poolTypeStr(poolType poolt){

        switch (poolt){
        case poolType::max: return "max";
        case poolType::avg: return "avg";
        default:            return "max";
        }
    }

    /// CPU, CUDA or OpenCL(for the future)
    enum class calcMode{
        CPU = 0,
        CUDA = 1,
        //OpenCL = 2,  
    };
    std::string calcModeStr(calcMode mode){

        switch (mode){
        case calcMode::CPU:  return "CPU";
        case calcMode::CUDA: return "CUDA";
        default:             return "CPU";
        }
    }

    enum class lockType{
        lock = 0,
        unlock = 1,
    };
    std::string lockTypeStr(lockType ltp){

        switch (ltp){
        case lockType::lock:   return "lock";
        case lockType::unlock: return "unlock";
        default:               return "unlock";
        }
    }

    enum class summatorType{
        summ = 0,
        diff = 1,
        mean = 2,
    };
    std::string summatorTypeStr(summatorType stp){

        switch (stp){
            case summatorType::summ: return "summ";
            case summatorType::diff: return "diff";
            case summatorType::mean: return "mean";
            default:                 return "summ";
        }
    }
        
    enum class lossType{
        softMaxToCrossEntropy = 0,
        binaryCrossEntropy = 1,
        regressionMSE = 2,
        userLoss = 3,
    };
    std::string lossTypeStr(lossType stp){

        switch (stp){
        case lossType::softMaxToCrossEntropy: return "softMaxToCrossEntropy";
        case lossType::binaryCrossEntropy: return "binaryCrossEntropy";
        case lossType::regressionMSE: return "regressionMSE";       ///< Mean Square Error
        case lossType::userLoss: return "userLoss";
        default:  return "userLoss";
        }
    }

    struct diap{

        uint32_t begin, end;
        diap(uint32_t begin_, uint32_t end_) :
            begin(begin_), end(end_){}
    };

    struct rect{
        uint32_t x, y, w, h;

        rect(uint32_t x_, uint32_t y_, uint32_t w_, uint32_t h_) :
            x(x_), y(y_), w(w_), h(h_){}
    };
}