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
    public interface IOperator
    {
        string getParamsJn();

        string name();
    } 
    
    /// <summary>
    ///  Input layer
    /// </summary>
    public class Input : IOperator
    {
        public string getParamsJn()
        {
                       
            return "{}";
        }

        public string name()
        {
            return "Input";
        }
    }

    /// <summary>
    ///  Fully connected layer
    /// </summary>
    public class FullyConnected : IOperator
    {
        public uint units;                                         ///< Number of out neurons. !Required parameter [0..)
        public active act = new active(active.type.relu);          ///< Activation function type. Optional parameter
        public optimizer opt = new optimizer(optimizer.type.adam); ///< Optimizer of weights. Optional parameter
        public float dropOut = 0.0f;                               ///< Random disconnection of neurons. Optional parameter [0..1.F]
        public batchNormType bnorm = new batchNormType(batchNormType.type.none); ///< Type of batch norm. Optional parameter
        public calcMode mode = new calcMode(calcMode.type.CPU);    ///< Сalculation mode. Optional parameter           
        public uint gpuDeviceId = 0;                               ///< GPU Id. Optional parameter
        public bool gpuClearMem = false;                           ///< Clear memory GPU. Optional parameter
        public bool freeze = false;                                ///< Do not change weights. Optional parameter
		public bool useBias = true;                                ///< +bias. Optional parameter
        public weightInit wini = new weightInit(weightInit.type.he); ///< Type of initialization of weights. Optional parameter
        public float decayMomentDW = 0.9F;                         ///< Optimizer of weights moment change. Optional parameter [0..1.F]
        public float decayMomentWGr = 0.99F;                       ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
        public float lmbRegular = 0.001F;                          ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
        public float batchNormLr = 0.001F;                         ///< Learning rate for batch norm coef. Optional parameter [0..)
        
        public FullyConnected(uint units_,                          
                       active.type act_ = active.type.relu,                
                       optimizer.type opt_ = optimizer.type.adam,          
                       float dropOut_ = 0.0f,                    
                       batchNormType.type bnorm_ = batchNormType.type.none,
                       calcMode.type mode_ = calcMode.type.CPU,            
                       uint gpuDeviceId_ = 0)
        {                                   
            units = units_;
            act = new active(act_);
            opt = new optimizer(opt_); 
            dropOut = dropOut_;
            bnorm = new batchNormType(bnorm_);
            mode = new calcMode(mode_);
            gpuDeviceId = gpuDeviceId_;
        }

        public FullyConnected(uint units_, calcMode.type mode_ = calcMode.type.CPU, 
            batchNormType.type bnorm_ = batchNormType.type.none)
        {        
            units = units_;            
            bnorm = new batchNormType(bnorm_);
            mode = new calcMode(mode_);  
        }

        public string getParamsJn()
        {

            string ss = "{\"units\":\"" + units.ToString() + "\"," +
                         "\"active\":\"" + act.str() + "\"," +
                         "\"weightInit\":\"" + wini.str() + "\"," +
                         "\"batchNorm\":\"" + bnorm.str() + "\"," +
                         "\"batchNormLr\":\"" + batchNormLr.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                         "\"optimizer\":\"" + opt.str() + "\"," +
                         "\"decayMomentDW\":\"" + decayMomentDW.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                         "\"decayMomentWGr\":\"" + decayMomentWGr.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                         "\"lmbRegular\":\"" + lmbRegular.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                         "\"dropOut\":\"" + dropOut.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                         "\"mode\":\"" + mode.str() + "\"," +
                         "\"gpuDeviceId\":\"" + gpuDeviceId.ToString() + "\"," +
                         "\"freeze\":\"" + (freeze ? "1" : "0") + "\"," +
						 "\"useBias\":\"" + (useBias ? "1" : "0") + "\"," +
                         "\"gpuClearMem\":\"" + (gpuClearMem ? "1" : "0") + "\"" +
                         "}";

           return ss;
        }

        public string name()
        {
            return "FullyConnected";
        }   
    }

    /// <summary>
    /// Convolution layer
    /// </summary>
    public class Convolution : IOperator
    {
                   
        public uint filters;                          ///< Number of output layers. !Required parameter [0..)
        public active act = new active(active.type.relu);          ///< Activation function type. Optional parameter
        public optimizer opt = new optimizer(optimizer.type.adam); ///< Optimizer of weights. Optional parameter
        public float dropOut = 0.0f;                  ///< Random disconnection of neurons. Optional parameter [0..1.F]
        public batchNormType bnorm = new batchNormType(batchNormType.type.none); ///< Type of batch norm. Optional parameter
        public uint fWidth = 3;                       ///< Width of mask. Optional parameter(> 0)
        public uint fHeight = 3;                      ///< Height of mask. Optional parameter(> 0)
        public int padding = 0;                       ///< Padding around the edges. Optional parameter
        public uint stride = 1;                       ///< Mask movement step. Optional parameter(> 0)
        public uint dilate = 1;                       ///< Expansion mask. Optional parameter(> 0)
        public calcMode mode = new calcMode(calcMode.type.CPU);        ///< Сalculation mode. Optional parameter           
        public uint gpuDeviceId = 0;                  ///< GPU Id. Optional parameter
        public bool gpuClearMem = false;              ///< Clear memory GPU. Optional parameter
        public bool freeze = false;                   ///< Do not change weights. Optional parameter
        public bool useBias = true;                   ///< +bias. Optional parameter
        public weightInit wini = new weightInit(weightInit.type.he);   ///< Type of initialization of weights. Optional parameter
        public float decayMomentDW = 0.9F;            ///< Optimizer of weights moment change. Optional parameter [0..1.F]
        public float decayMomentWGr = 0.99F;          ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
        public float lmbRegular = 0.001F;             ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
        public float batchNormLr = 0.001F;            ///< Learning rate for batch norm coef. Optional parameter [0..)
        
        public Convolution(uint filters_,              
            active.type act_ = active.type.relu,                
            optimizer.type opt_ = optimizer.type.adam,          
            float dropOut_ = 0.0F,                    
            batchNormType.type bnorm_ = batchNormType.type.none,
            uint fWidth_ = 3,                      
            uint fHeight_ = 3,                    
            int padding_ = 0,
            uint stride_ = 1,                      
            uint dilate_ = 1,                      
            calcMode.type mode_ = calcMode.type.CPU,            
            uint gpuDeviceId_ = 0)
            {
                        
            filters = filters_;
            act = new active(act_);
            opt = new optimizer(opt_);
            dropOut  =dropOut_;
            bnorm = new batchNormType(bnorm_);
            fWidth = fWidth_;
            fHeight = fHeight_;
            padding  = padding_;
            stride = stride_;
            dilate = dilate_;
            mode = new calcMode(mode_);
            gpuDeviceId = gpuDeviceId_;
        }
       
        public Convolution(uint filters_, int padding_ = 0, calcMode.type mode_ = calcMode.type.CPU, 
            batchNormType.type bnorm_ = batchNormType.type.none)
        {
            filters = filters_;
            padding = padding_;
            mode = new calcMode(mode_);
            bnorm = new batchNormType(bnorm_);
        }

        public string getParamsJn()
        {

            string ss = "{\"filters\":\"" + filters.ToString() + "\"," +
                "\"fWidth\":\"" + fWidth.ToString() + "\"," +
                "\"fHeight\":\"" + fHeight.ToString() + "\"," +
                "\"padding\":\"" + padding.ToString() + "\"," +
                "\"stride\":\"" + stride.ToString() + "\"," +
                "\"dilate\":\"" + dilate.ToString() + "\"," +
                "\"active\":\"" + act.str() + "\"," +
                "\"weightInit\":\"" + wini.str() + "\"," +
                "\"batchNorm\":\"" + bnorm.str() + "\"," +
                "\"batchNormLr\":\"" + batchNormLr.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                "\"optimizer\":\"" + opt.str() + "\"," +
                "\"decayMomentDW\":\"" + decayMomentDW.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                "\"decayMomentWGr\":\"" + decayMomentWGr.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                "\"lmbRegular\":\"" + lmbRegular.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                "\"dropOut\":\"" + dropOut.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                "\"mode\":\"" + mode.str() + "\"," +
                "\"gpuDeviceId\":\"" + gpuDeviceId.ToString() + "\"," +
                "\"freeze\":\"" + (freeze ? "1" : "0") + "\"," +
				"\"useBias\":\"" + (useBias ? "1" : "0") + "\"," +
                "\"gpuClearMem\":\"" + (gpuClearMem ? "1" : "0") + "\"" +
                "}";

            return ss;
        }

        public string name()
        {
            return "Convolution";
        }
          
    };

    /// <summary>
    /// Deconvolution layer
    /// </summary>
    public class Deconvolution : IOperator
    {

        public uint filters;                          ///< Number of output layers. !Required parameter [0..)
        public active act = new active(active.type.relu);          ///< Activation function type. Optional parameter
        public optimizer opt = new optimizer(optimizer.type.adam); ///< Optimizer of weights. Optional parameter
        public float dropOut = 0.0f;                  ///< Random disconnection of neurons. Optional parameter [0..1.F]
        public batchNormType bnorm = new batchNormType(batchNormType.type.none); ///< Type of batch norm. Optional parameter
        public uint fWidth = 3;                       ///< Width of mask. Optional parameter(> 0)
        public uint fHeight = 3;                      ///< Height of mask. Optional parameter(> 0)
        public uint stride = 2;                       ///< Mask movement step. Optional parameter(> 0)
        public calcMode mode = new calcMode(calcMode.type.CPU);        ///< Сalculation mode. Optional parameter           
        public uint gpuDeviceId = 0;                  ///< GPU Id. Optional parameter
        public bool gpuClearMem = false;              ///< Clear memory GPU. Optional parameter
        public bool freeze = false;                   ///< Do not change weights. Optional parameter
        public weightInit wini = new weightInit(weightInit.type.he);   ///< Type of initialization of weights. Optional parameter
        public float decayMomentDW = 0.9F;            ///< Optimizer of weights moment change. Optional parameter [0..1.F]
        public float decayMomentWGr = 0.99F;          ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
        public float lmbRegular = 0.001F;             ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
        public float batchNormLr = 0.001F;            ///< Learning rate for batch norm coef. Optional parameter [0..)

        public Deconvolution(uint filters_,
            active.type act_ = active.type.relu,
            optimizer.type opt_ = optimizer.type.adam,
            float dropOut_ = 0.0F,
            batchNormType.type bnorm_ = batchNormType.type.none,
            uint fWidth_ = 3,
            uint fHeight_ = 3,
            uint stride_ = 1,
            calcMode.type mode_ = calcMode.type.CPU,
            uint gpuDeviceId_ = 0)
        {

            filters = filters_;
            act = new active(act_);
            opt = new optimizer(opt_);
            dropOut = dropOut_;
            bnorm = new batchNormType(bnorm_);
            fWidth = fWidth_;
            fHeight = fHeight_;
            stride = stride_;
            mode = new calcMode(mode_);
            gpuDeviceId = gpuDeviceId_;
        }

        public Deconvolution(uint filters_, calcMode.type mode_ = calcMode.type.CPU,
            batchNormType.type bnorm_ = batchNormType.type.none)
        {
            filters = filters_;
            mode = new calcMode(mode_);
            bnorm = new batchNormType(bnorm_);
        }

        public string getParamsJn()
        {

            string ss = "{\"filters\":\"" + filters.ToString() + "\"," +
                "\"fWidth\":\"" + fWidth.ToString() + "\"," +
                "\"fHeight\":\"" + fHeight.ToString() + "\"," +
                "\"stride\":\"" + stride.ToString() + "\"," +
                "\"active\":\"" + act.str() + "\"," +
                "\"weightInit\":\"" + wini.str() + "\"," +
                "\"batchNorm\":\"" + bnorm.str() + "\"," +
                "\"batchNormLr\":\"" + batchNormLr.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                "\"optimizer\":\"" + opt.str() + "\"," +
                "\"decayMomentDW\":\"" + decayMomentDW.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                "\"decayMomentWGr\":\"" + decayMomentWGr.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                "\"lmbRegular\":\"" + lmbRegular.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                "\"dropOut\":\"" + dropOut.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture) + "\"," +
                "\"mode\":\"" + mode.str() + "\"," +
                "\"gpuDeviceId\":\"" + gpuDeviceId.ToString() + "\"," +
                "\"freeze\":\"" + (freeze ? "1" : "0") + "\"," +
                "\"gpuClearMem\":\"" + (gpuClearMem ? "1" : "0") + "\"" +
                "}";

            return ss;
        }

        public string name()
        {
            return "Deconvolution";
        }

    };

    /// <summary>
    ///  Pooling layer
    /// </summary>
    public class Pooling : IOperator
    {

        public uint kernel = 2;              ///< Square Mask Size. Optional parameter (> 0)
        public uint stride = 2;              ///< Mask movement step. Optional parameter(> 0)		
        public pooling pool = new pooling(pooling.type.max);    ///< Operator Type. Optional parameter 
        public calcMode mode = new calcMode(calcMode.type.CPU); ///< Сalculation mode. Optional parameter           
        public uint gpuDeviceId = 0;         ///< GPU Id. Optional parameter
        public bool gpuClearMem = false;     ///< Clear memory GPU. Optional parameter

        public Pooling(calcMode.type mode_ = calcMode.type.CPU,
            uint gpuDeviceId_ = 0,
            bool gpuClearMem_ = false)
        {
            mode = new calcMode(mode_);
            gpuDeviceId = gpuDeviceId_;
            gpuClearMem = gpuClearMem_;
        }

        public string getParamsJn()
        {

            string ss = "{\"kernel\":\"" + kernel.ToString() + "\"," +
			             "\"stride\":\"" + stride.ToString() + "\"," +
                         "\"pool\":\"" + pool.str() + "\"," +
                         "\"mode\":\"" + mode.str() + "\"," +
                         "\"gpuDeviceId\":\"" + gpuDeviceId.ToString() + "\"," +
                         "\"gpuClearMem\":\"" + (gpuClearMem ? "1" : "0") + "\"" +
                         "}";

            return ss;
        }

        public string name()
        {
            return "Pooling";
        }
    };

    /// <summary>
    /// Operator to block further calculation at the current location.
    /// It is designed for the ability to dynamically disconnect the parallel
    /// branches of the network during operation.
    /// </summary>
    public class Lock : IOperator
    {

        public lockType lockTp;    ///< Blocking activity. Optional parameter

        public Lock(lockType.type lockTp_)
        {
           lockTp= new lockType(lockTp_);
        }

        public string getParamsJn()
        {                                
            string ss = "{\"state\":\"" + lockTp.str() + "\"}";

            return ss;
        }

        public string name()
        {
            return "Lock";
        }
    };

    /// <summary>
    /// Operator for transferring data to several nodes at once.
    /// Data can only be received from one node.
    /// </summary>
    public class Switch : IOperator
    {

        public string nextWay;   // next nodes through a space

        public Switch(string nextWay_)
        {
            nextWay = nextWay_;
        }

        public string getParamsJn()
        {

            string ss = "{\"nextWay\":\"" + nextWay + "\"}";

            return ss;
        }

        public string name()
        {
            return "Switch";
        }
    };

    /// <summary>
    ///  The operator is designed to combine the values of two layers.
    /// The consolidation can be performed by the following options: "summ", "diff", "mean".
    /// The dimensions of the input layers must be the same.
    /// </summary>
    public class Summator : IOperator
    {
        
        public summatorType summType;     

        public Summator(summatorType.type summType_ = summatorType.type.summ)
        {
            summType = new summatorType(summType_);
        }

        public string getParamsJn()
        {

            string ss = "{\"type\":\"" + summType.str() + "\"}";

            return ss;
        }

        public string name()
        {
            return "Summator";
        }
    };

    /// <summary>
    ///  The operator connects the channels with multiple layers
    /// </summary>
    public class Concat : IOperator
    {
                      
        public string sequence;    // prev nodes through a space

        public Concat(string sequence_){
            
            sequence = sequence_;
        } 
        
        public string getParamsJn(){

            string ss = "{\"sequence\":\"" + sequence + "\"}";

            return ss;
        }

        public string name()
        {
            return "Concat";
        }
    };

    /// <summary>
    ///  Change the number of channels
    /// </summary>
    public class Resize : IOperator
    {
               
        public diap fwdDiap, bwdDiap;   // diap layer through a space

        public Resize(diap fwdDiap_, diap bwdDiap_){

            fwdDiap = fwdDiap_;
            bwdDiap = bwdDiap_;
        }
      
        public string getParamsJn(){
                       
           string ss = "{\"fwdDiap\":\"" + fwdDiap.begin.ToString() + " " + fwdDiap.end.ToString() + "\"," +
                "\"bwdDiap\":\"" + bwdDiap.begin.ToString() + " " .ToString() + bwdDiap.end.ToString() + "\"}";

            return ss;
        }

        public string name()
        {
            return "Resize";
        }
    };

    /// <summary>
    /// ROI clipping in each image of each channel
    /// </summary>
    public class Crop : IOperator
    {
                
        public rect rct;         // region of interest

        public Crop(rect rct_){
            rct = rct_;
        }

        public string getParamsJn()
        {

             string ss = "{\"roi\":\"" +
                 rct.x.ToString() + " " + rct.y.ToString() + " " + rct.w.ToString() + " " + rct.h.ToString() + "\"}";

            return ss;
        }

        public string name()
        {
            return "Crop";
        }
    };

    /// <summary>
    /// Activation function
    /// </summary>
    public class Activation : IOperator
    {

        public active act = new active(active.type.relu);          ///< Activation function type. Optional parameter

        public Activation(active act_)
        {
            act = act_;
        }

        public string getParamsJn()
        {

            string ss = "{\"active\":\"" + act.str() + "\"}";

            return ss;
        }

        public string name()
        {
            return "Activation";
        }
    };

    /// <summary>
    /// Batch norm
    /// </summary>
    public class BatchNormLayer : IOperator
    {
        public batchNormType bnorm = new batchNormType(batchNormType.type.byChannels); ///< Type of batch norm. Optional parameter
       
        public BatchNormLayer(batchNormType bnorm_)
        {
            bnorm = bnorm_;
        }

        public string getParamsJn()
        {                       
            return "{\"bnType\":\"" + bnorm.str() + "\"}";
        }

        public string name()
        {
            return "BatchNorm";
        }        
    };

     /// <summary>
     /// Custom layer
     /// </summary>
    public class UserLayer : IOperator
    {
              
        public string cbackName;

        public UserLayer(string cbackName_)
        {
            cbackName = cbackName_;
        }

        public string getParamsJn()
        {
            
            string ss = "{\"cbackName\":\"" + cbackName + "\"}";

            return ss;
        }

        public string name()
        {
            return "UserLayer";
        }
    };
    
    /// <summary>
    /// Error function calculation layer
    /// </summary>
    public class LossFunction : IOperator
    {
                   
        public lossType loss;

        public LossFunction(lossType.type loss_)
        {
            loss = new lossType(loss_);
        }
    
        public string getParamsJn(){

            string ss =  "{\"loss\":\"" + loss.str() + "\"}";

            return ss;
        }

        public string name(){
            return "LossFunction";
        }
    };
}
