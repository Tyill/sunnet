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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace SN_API
{
    /// <summary>
    /// Activation function type
    /// </summary>
    public struct active{

        public active(type tp)
        {
            type_ = tp;
        }

        /// Activation function type
        public enum type{
            none = -1,
            sigmoid = 0,
            relu = 1,
            leakyRelu = 2,
            elu = 3,
        }

        public string str()
        {
            switch (type_)
            {
                case type.none:      return "none";
                case type.sigmoid:   return "sigmoid";
                case type.relu:      return "relu";
                case type.leakyRelu: return "leakyRelu";
                case type.elu:       return "elu";
                default:             return "none";
            }
        }

        private type type_;
    };

    /// <summary>
    /// Type of initialization of weights
    /// </summary>
    public struct weightInit{

        public weightInit(type tp)
        {
            type_ = tp;
        }

        public enum type
        {
            uniform = 0,
            he = 1,
            lecun = 2,
            xavier = 3,
        };
        public string str()
        {
            switch (type_)
            {
                case type.uniform: return "uniform";
                case type.he:      return "he";
                case type.lecun:   return "lecun";
                case type.xavier:  return "xavier";
                default:           return "uniform";
            }
        }
        private type type_;
    }

    /// <summary>
    /// Type of batch norm 
    /// </summary>
    public struct batchNormType{

        public batchNormType(type tp)
        {
            type_ = tp;
        }

        public enum type
        {
            none = -1,
            beforeActive = 0,
            postActive = 1,
        };
        public string str()
        {
            switch (type_)
            {
                case type.none:         return "none";
                case type.beforeActive: return "beforeActive";
                case type.postActive:   return "postActive";
                default:                return "none";
            }
        }
        private type type_;
    }

    /// <summary>
    /// Optimizer of weights
    /// </summary>
    public struct optimizer{

        public optimizer(type tp)
        {
            type_ = tp;
        }

        public enum type
        {
            sgd = 0,
            sgdMoment = 1,
            adagrad = 2,
            RMSprop = 3,
            adam = 4,
        };
        public string str()
        {
            switch (type_)
            {
            case type.sgd:       return "sgd";
            case type.sgdMoment: return "sgdMoment";
            case type.adagrad:   return "adagrad";
            case type.RMSprop:   return "RMSprop";
            case type.adam:      return "adam";
            default:             return "adam";
            }
        }
        private type type_;
    }
    
    /// <summary>
    /// pooling
    /// </summary>
    public struct pooling{

        public pooling(type tp)
        {
            type_ = tp;
        }

        public enum type
        {
            max = 0,
            avg = 1,
        };
        public string str()
        {
            switch (type_)
            {
            case type.max: return "max";
            case type.avg: return "avg";
            default:       return "max";
            }
        }
        private type type_;
    }

    /// <summary>
    /// CPU, CUDA or OpenCL
    /// </summary>
    public struct calcMode{

        public calcMode(type tp)
        {
            type_ = tp;
        }
        
        public enum type
        {
            CPU = 0,
            CUDA = 1,
            OpenCL = 2,  
        };
        public string str()
        {
            switch (type_)
            {
            case type.CPU:    return "CPU";
            case type.CUDA:   return "CUDA";
            case type.OpenCL: return "OpenCL";
            default:          return "CPU";
            }
        }
        private type type_;
    }

    /// <summary>
    /// lockType
    /// </summary>
    public struct lockType{

        public lockType(type tp)
        {
            type_ = tp;
        }

        public enum type
        {
            tlock = 0,
            tunlock = 1,
        };
        public string str()
        {
            switch (type_)
            {
            case type.tlock:   return "lock";
            case type.tunlock: return "unlock";
            default:          return "unlock";
            }
        }
        private type type_;
    }

    /// <summary>
    /// summatorType
    /// </summary>
    public struct summatorType{

        public summatorType(type tp)
        {
            type_ = tp;
        }
                
        public enum type
        {
            summ = 0,
            diff = 1,
            mean = 2,
        };
        public string str()
        {
            switch (type_)
            {
                case type.summ: return "summ";
                case type.diff: return "diff";
                case type.mean: return "mean";
                default:        return "summ";
            }
        }
        private type type_;
    }
    
    /// <summary>
    /// lossType
    /// </summary>
    public struct lossType{

        public lossType(type tp)
        {
            type_ = tp;
        }
    
        public enum type
        {
            softMaxToCrossEntropy = 0,
            binaryCrossEntropy = 1,
            regressionMSE = 2,
            userLoss = 3,
        };
        public string str()
        {
            switch (type_){
            case type.softMaxToCrossEntropy: return "softMaxToCrossEntropy";
            case type.binaryCrossEntropy:    return "binaryCrossEntropy";
            case type.regressionMSE:         return "regressionMSE";       ///< Mean Square Error
            case type.userLoss:              return "userLoss";
            default:                         return "userLoss";
            }
        }
        private type type_;
    }

    /// <summary>
    /// diap
    /// </summary>
    public struct diap{

        public uint begin, end;
        public diap(uint begin_, uint end_)
        {
            begin = begin_; end = end_;
        }
    };

    /// <summary>
    /// rect
    /// </summary>
    public struct rect{
        public uint x, y, w, h;

        public rect(uint x_, uint y_, uint w_, uint h_)
        {
            x = x_; y = y_; w = w_; h = h_;
        }
    };
}
