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
#include "snOperatorCUDA/snOperator.h"
#include "snBase/snBase.h"
#include "Operator/fullyConnected.h"
#include "Operator/convolution.h"
#include "Operator/deconvolution.h"
#include "Operator/pooling.h"
#include "Operator/input.h"
#include "Operator/output.h"
#include "Operator/lossFunction.h"
#include "Operator/lock.h"
#include "Operator/summator.h"
#include "Operator/userLayer.h"
#include "Operator/switch.h"
#include "Operator/crop.h"
#include "Operator/concat.h"
#include "Operator/resize.h"
#include "Operator/batchNorm.h"
#include "Operator/activation.h"

namespace SN_Opr{

    SN_Base::OperatorBase* createOperator(void* net, const std::string& fname, const std::string& node,
        std::map<std::string, std::string>& prms){

        SN_Base::OperatorBase* ret = nullptr;

        if (fname == "Input")               ret = (SN_Base::OperatorBase*)new Input(net, fname, node, prms);
        else if (fname == "Output")         ret = (SN_Base::OperatorBase*)new Output(net, fname, node, prms);
        else if (fname == "FullyConnected") ret = (SN_Base::OperatorBase*)new FullyConnected(net, fname, node, prms);
        else if (fname == "LossFunction")   ret = (SN_Base::OperatorBase*)new LossFunction(net, fname, node, prms);
        else if (fname == "Convolution")    ret = (SN_Base::OperatorBase*)new Convolution(net, fname, node, prms);
        else if (fname == "Deconvolution")  ret = (SN_Base::OperatorBase*)new Deconvolution(net, fname, node, prms);
        else if (fname == "Pooling")        ret = (SN_Base::OperatorBase*)new Pooling(net, fname, node, prms);
        else if (fname == "Lock")           ret = (SN_Base::OperatorBase*)new Lock(net, fname, node, prms);
        else if (fname == "Summator")       ret = (SN_Base::OperatorBase*)new Summator(net, fname, node, prms);
        else if (fname == "Switch")         ret = (SN_Base::OperatorBase*)new Switch(net, fname, node, prms);
        else if (fname == "UserLayer")      ret = (SN_Base::OperatorBase*)new UserLayer(net, fname, node, prms);
        else if (fname == "Crop")           ret = (SN_Base::OperatorBase*)new Crop(net, fname, node, prms);
        else if (fname == "Concat")         ret = (SN_Base::OperatorBase*)new Concat(net, fname, node, prms);
        else if (fname == "Resize")         ret = (SN_Base::OperatorBase*)new Resize(net, fname, node, prms);
        else if (fname == "BatchNorm")      ret = (SN_Base::OperatorBase*)new BatchNorm(net, fname, node, prms);
        else if (fname == "Activation")     ret = (SN_Base::OperatorBase*)new Activation(net, fname, node, prms);

        return ret;
    }
    
    void freeOperator(SN_Base::OperatorBase* opr, const std::string& fname){

        if (opr){
            if (fname == "Input")               delete (Input*)opr;
            else if (fname == "Output")         delete (Output*)opr;
            else if (fname == "FullyConnected") delete (FullyConnected*)opr;
            else if (fname == "LossFunction")   delete (LossFunction*)opr;
            else if (fname == "Convolution")    delete (Convolution*)opr;
            else if (fname == "Deconvolution")  delete (Deconvolution*)opr;
            else if (fname == "Pooling")        delete (Pooling*)opr;
            else if (fname == "Lock")           delete (Lock*)opr;
            else if (fname == "Summator")       delete (Summator*)opr;
            else if (fname == "Switch")         delete (Switch*)opr;
            else if (fname == "UserLayer")      delete (UserLayer*)opr;
            else if (fname == "Crop")           delete (Crop*)opr;
            else if (fname == "Concat")         delete (Concat*)opr;
            else if (fname == "Resize")         delete (Resize*)opr;
            else if (fname == "BatchNorm")      delete (BatchNorm*)opr;
            else if (fname == "Activation")     delete (Activation*)opr;
        }
    }
}
