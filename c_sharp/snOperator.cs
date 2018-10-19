using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace SN_API
{
    /// <summary>
    ///  Input layer
    /// </summary>
    public class Input
    {   
        string getParamsJn(){
                       
            return "{}";
        }

        string name(){
            return "Input";
        }
    }

    /// <summary>
    ///  Fully connected layer
    /// </summary>
    public class FullyConnected
    {
        public uint kernel;                                        ///< Number of hidden neurons. !Required parameter [0..)
        public active act = new active(active.type.relu);          ///< Activation function type. Optional parameter
        public optimizer opt = new optimizer(optimizer.type.adam); ///< Optimizer of weights. Optional parameter
        public float dropOut = 0.0f;                               ///< Random disconnection of neurons. Optional parameter [0..1.F]
        public batchNormType bnorm = new batchNormType(batchNormType.type.none); ///< Type of batch norm. Optional parameter
        public calcMode mode = new calcMode(calcMode.type.CPU);    ///< Сalculation mode. Optional parameter           
        public uint gpuDeviceId = 0;                               ///< GPU Id. Optional parameter
        public bool gpuClearMem = false;                           ///< Clear memory GPU. Optional parameter
        public bool freeze = false;                                ///< Do not change weights. Optional parameter
        public weightInit wini = new weightInit(weightInit.type.he); ///< Type of initialization of weights. Optional parameter
        public float decayMomentDW = 0.9F;                         ///< Optimizer of weights moment change. Optional parameter [0..1.F]
        public float decayMomentWGr = 0.99F;                       ///< Optimizer of weights moment change of prev. Optional parameter [0..1.F]
        public float lmbRegular = 0.001F;                          ///< Optimizer of weights l2Norm. Optional parameter [0..1.F]
        public float batchNormLr = 0.001F;                         ///< Learning rate for batch norm coef. Optional parameter [0..)
        
        FullyConnected(uint kernel_,                          
                       active.type act_ = active.type.relu,                
                       optimizer.type opt_ = optimizer.type.adam,          
                       float dropOut_ = 0.0f,                    
                       batchNormType.type bnorm_ = batchNormType.type.none,
                       calcMode.type mode_ = calcMode.type.CPU,            
                       uint gpuDeviceId_ = 0)
        {                                   
            kernel = kernel_;
            act = new active(act_);
            opt = new optimizer(opt_); 
            dropOut = dropOut_;
            bnorm = new batchNormType(bnorm_);
            mode = new calcMode(mode_);
            gpuDeviceId = gpuDeviceId_;
        }

        public FullyConnected(uint kernel_, calcMode.type mode_ = calcMode.type.CPU, 
            batchNormType.type bnorm_ = batchNormType.type.none)
        {        
            kernel = kernel_;            
            bnorm = new batchNormType(bnorm_);
            mode = new calcMode(mode_);  
        }
                      
        string getParamsJn(){

            string ss = "{\"kernel\":\"" + kernel.ToString() + "\"," +
                         "\"active\":\"" + act.str() + "\"," +
                         "\"weightInit\":\"" + wini.str() + "\"," +
                         "\"batchNorm\":\"" + bnorm.str() + "\"," +
                         "\"batchNormLr\":\"" + batchNormLr.ToString() + "\"," +
                         "\"optimizer\":\"" + opt.str() + "\"," +
                         "\"decayMomentDW\":\"" + decayMomentDW.ToString() + "\"," +
                         "\"decayMomentWGr\":\"" + decayMomentWGr.ToString() + "\"," +
                         "\"lmbRegular\":\"" + lmbRegular.ToString() + "\"," +
                         "\"dropOut\":\"" + dropOut.ToString() + "\"," +
                         "\"mode\":\"" + mode.str() + "\"," +
                         "\"gpuDeviceId\":\"" + gpuDeviceId.ToString() + "\"," +
                         "\"freeze\":\"" + (freeze ? "1" : "0") + "\"," +
                         "\"gpuClearMem\":\"" + (gpuClearMem ? "1" : "0") + "\"" +
                         "}";

           return ss;
        }

        string name(){
            return "FullyConnected";
        }   
    }
}
