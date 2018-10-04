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

#include "stdafx.h"
#include "snBase/snBase.h"
#include "skynet/skynet.h"
#include "snet.h"


namespace SN_API{

    /// create net
    /// @param[in] jnNet - network architecture in JSON
    /// @param[out] out_err - parse error jnNet. "" - ok. The memory is allocated by the user
    /// @param[in] statusCBack - callback state. Not necessary
    /// @param[in] udata - user data. Not necessary
    skyNet snCreateNet(const char* jnNet,
        char* out_err /*sz 256*/,
        snStatusCBack sts,
        snUData ud){

        if (!jnNet || !out_err) return nullptr;

        auto net = new SNet(jnNet, out_err, sts, ud);

        if (strlen(out_err) > 0){
            delete net;
            net = nullptr;
        }

        return net;
    }
    
    /// get last error
    /// @param[in] skyNet - object net
    /// @param[out] out_err - parse error jnNet. "" - ok. The memory is allocated by the user
    void snGetLastErrorStr(skyNet fn, char* out_err){

        if (!fn || !out_err) return;

        static_cast<SNet*>(fn)->getLastErrorStr(out_err);
    }

    /// training - a cycle forward-back with auto-correction of weights
    /// @param[in] skyNet - object net
    /// @param[in] lr - learning rate
    /// @param[in] isz - input layer size
    /// @param[in] iLayer - input layer        
    /// @param[in] osz - size of target and result. Sets for verification
    /// @param[out] outData - result, the size must match the markup. The memory is allocated by the user
    /// @param[in] targetData - target, the size must match the markup. The memory is allocated by the user
    /// @param[out] outAccurate - current accuracy
    /// @return true - ok
    bool snTraining(skyNet fn,
                    snFloat lr,
                    snLSize isz,
                    const snFloat* iLayer,
                    snLSize osz,
                    snFloat* outData,
                    const snFloat* targetData,
                    snFloat* outAccurate){

        if (!fn) return false;

        if (!iLayer || !outData || !targetData || !outAccurate){
            static_cast<SNet*>(fn)->statusMess("SN error: !iLayer || !outData || !targetData || !outAccurate");
            return false;
        }

        SN_Base::snSize bsz(isz.w, isz.h, isz.ch, isz.bsz);
        SN_Base::snSize tnsz(osz.w, osz.h, osz.ch, osz.bsz);

        if ((bsz.size() == 0) || (tnsz.size() == 0)){
            static_cast<SNet*>(fn)->statusMess("SN error: (isz == 0) || (osz == 0)");
            return false;
        }

        return static_cast<SNet*>(fn)->training(lr, bsz, iLayer, tnsz, outData, targetData, outAccurate);
    }

    /// forward pass
    /// @param[in] skyNet - object net
    /// @param[in] isLern - is lern?
    /// @param[in] isz - input layer size
    /// @param[in] iLayer - input layer       
    /// @param[in] osz - size of result. Sets for verification
    /// @param[out] outData - result, the size must match the markup. The memory is allocated by the user
    /// @return true - ok
    bool snForward(skyNet fn,
                   bool isLern,
                   snLSize isz,
                   const snFloat* iLayer,
                   snLSize osz,
                   snFloat* outData){

        if (!fn) return false;

        if (!iLayer || !outData){
            static_cast<SNet*>(fn)->statusMess("SN error: !iLayer || !outData");
            return false;
        }

        SN_Base::snSize bsz(isz.w, isz.h, isz.ch, isz.bsz);
        SN_Base::snSize onsz(osz.w, osz.h, osz.ch, osz.bsz);

        if ((bsz.size() == 0) || (onsz.size() == 0)){
            static_cast<SNet*>(fn)->statusMess("SN error: (isz == 0) || (osz == 0)");
            return false;
        }

        return static_cast<SNet*>(fn)->forward(isLern, bsz, iLayer, onsz, outData);
    }

    /// backward pass
    /// @param[in] skyNet - object net
    /// @param[in] lr - learning rate
    /// @param[in] gsz - size of the error gradient. Sets for verification
    /// @param[in] grad - error gradient, the size must match the output
    /// @return true - ok
    bool snBackward(skyNet fn,
                    snFloat lr,
                    snLSize gsz,
                    const snFloat* grad){

        if (!fn) return false;

        if (!grad){
            static_cast<SNet*>(fn)->statusMess("SN error: !grad");
            return false;
        }

        SN_Base::snSize gnsz(gsz.w, gsz.h, gsz.ch, gsz.bsz);

        if (gnsz.size() == 0){
            static_cast<SNet*>(fn)->statusMess("SN error: gsz == 0");
            return false;
        }

        return static_cast<SNet*>(fn)->backward(lr, gnsz, grad);
    }

    /// set weight of node
    /// @param[in] skyNet - object net
    /// @param[in] nodeName - name node
    /// @param[in] wsz - size
    /// @param[in] wData - weight        
    /// @return true - ok
    bool snSetWeightNode(skyNet fn, const char* nodeName, snLSize wsz, const snFloat* wData){

        if (!fn) return false;

        if (!nodeName || !wData){
            static_cast<SNet*>(fn)->statusMess("SN error: !nodeName || !wData");
            return false;
        }

        SN_Base::snSize bsz(wsz.w, wsz.h, wsz.ch, wsz.bsz);

        if (bsz.size() == 0){
            static_cast<SNet*>(fn)->statusMess("SN error: wsz == 0");
            return false;
        }

        return static_cast<SNet*>(fn)->setWeightNode(nodeName, bsz, wData);
    }

    /// get weight of node
    /// @param[in] skyNet - object net
    /// @param[in] nodeName - name node
    /// @param[out] wsz - output size
    /// @param[out] wData - output data. First pass NULL, then pass it to the same 
    /// @return true - ok
    bool snGetWeightNode(skyNet fn, const char* nodeName, snLSize* wsz, snFloat** wData){

        if (!fn) return false;

        if (!nodeName || !wsz){
            static_cast<SNet*>(fn)->statusMess("SN error: !nodeName || !wsz");
            return false;
        }

        SN_Base::snSize bsz;
        if (!static_cast<SNet*>(fn)->getWeightNode(nodeName, bsz, wData)) return false;

        wsz->w = bsz.w;
        wsz->h = bsz.h;
        wsz->ch = bsz.d;
        wsz->bsz = bsz.n;
                
        return true;
    }

    /// set batchNorm of node
    /// @param[in] skyNet - object net
    /// @param[in] nodeName - name node
    /// @param[in] bnsz - data size
    /// @param[in] bnData - data       
    /// @return true - ok
    bool snSetBatchNormNode(skyNet fn, const char* nodeName, snLSize bnsz, SN_API::batchNorm bnData){

        if (!fn) return false;

        if (!nodeName || !bnData.mean || !bnData.varce || !bnData.scale || !bnData.schift){
            static_cast<SNet*>(fn)->statusMess("SN error: !nodeName || !bnData.mean || !bnData.varce || !bnData.scale || !bnData.schift");
            return false;
        }

        SN_Base::batchNorm bn;        
        bn.mean = bnData.mean;
        bn.varce = bnData.varce;
        bn.scale = bnData.scale;
        bn.schift = bnData.schift;
        bn.sz = SN_Base::snSize(bnsz.w, bnsz.h, bnsz.ch);

        if (bn.sz.size() == 0){
            static_cast<SNet*>(fn)->statusMess("SN error: bnsz == 0");
            return false;
        }

        return static_cast<SNet*>(fn)->setBatchNormNode(nodeName, bn);
    }

    /// get batchNorm of node
    /// @param[in] skyNet - object net
    /// @param[in] nodeName - name node
    /// @param[out] bnsz - data size
    /// @param[out] bnData - data         
    /// @return true - ok
    bool snGetBatchNormNode(skyNet fn, const char* nodeName, snLSize* bnsz, batchNorm* bnData){

        if (!fn) return false;

        if (!nodeName || !bnsz || !bnData){
            static_cast<SNet*>(fn)->statusMess("SN error: !nodeName || !bnsz || !bnData");
            return false;
        }

        SN_Base::batchNorm bn;
        if (!static_cast<SNet*>(fn)->getBatchNormNode(nodeName, bn)) return false;

        size_t sz = bn.sz.size();
        if (sz == 0){
            static_cast<SNet*>(fn)->statusMess("SN error: bnorm not found");
            return false;
        }

        bnData->mean =   new snFloat[sz]; memcpy(bnData->mean,  bn.mean,  sz * sizeof(snFloat));
        bnData->varce =  new snFloat[sz]; memcpy(bnData->varce, bn.varce, sz * sizeof(snFloat));
        bnData->scale =  new snFloat[sz]; memcpy(bnData->scale, bn.scale, sz * sizeof(snFloat));
        bnData->schift = new snFloat[sz]; memcpy(bnData->schift,bn.schift,sz * sizeof(snFloat));

        *bnsz = snLSize(bn.sz.w, bn.sz.h, bn.sz.d);

        return true;
    }
    
    /// set input node (relevant for additional inputs)
    /// @param[in] skyNet - object net
    /// @param[in] nodeName - name node
    /// @param[in] isz - data size
    /// @param[in] inData - data       
    /// @return true - ok
    bool snSetInputNode(skyNet fn,
                        const char* nodeName,
                        snLSize isz,
                        const snFloat* inData){

        if (!fn) return false;

        if (!nodeName || !inData){
            static_cast<SNet*>(fn)->statusMess("SN error: !nodeName || !inData");
            return false;
        }

        SN_Base::snSize bsz(isz.w, isz.h, isz.ch, isz.bsz);

        if (bsz.size() == 0){
            static_cast<SNet*>(fn)->statusMess("SN error: isz == 0");
            return false;
        }

        return static_cast<SNet*>(fn)->setInputNode(nodeName, bsz, inData);
    }

    /// get output node (relevant for additional inputs)
    /// @param[in] skyNet - object net
    /// @param[in] nodeName - name node
    /// @param[out] osz - data size
    /// @param[out] outData - data. First pass NULL, then pass it to the same 
    /// @return true - ok
    bool snGetOutputNode(skyNet fn,
                         const char* nodeName,
                         snLSize* osz,
                         snFloat** outData){

        if (!fn) return false;

        if (!nodeName || !osz){
            static_cast<SNet*>(fn)->statusMess("SN error: !nodeName || !osz");
            return false;
        }

        SN_Base::snSize bsz;
        if (!static_cast<SNet*>(fn)->getOutputNode(nodeName, bsz, outData)) return false;

        osz->w = bsz.w;
        osz->h = bsz.h;
        osz->ch = bsz.d;
        osz->bsz = bsz.n;

        return true;

    }

    /// set gradient node (relevant for additional outputs)
    /// @param[in] skyNet - object net
    /// @param[in] nodeName - name node
    /// @param[in] gsz - data size
    /// @param[in] gData - data        
    /// @return true - ok
    bool snSetGradientNode(skyNet fn,
        const char* nodeName,
        snLSize gsz,
        const snFloat* gData){

        if (!fn) return false;

        if (!nodeName || !gData){
            static_cast<SNet*>(fn)->statusMess("SN error: !nodeName || !gData");
            return false;
        }

        SN_Base::snSize bsz(gsz.w, gsz.h, gsz.ch, gsz.bsz);

        if (bsz.size() == 0){
            static_cast<SNet*>(fn)->statusMess("SN error: gsz == 0");
            return false;
        }

        return static_cast<SNet*>(fn)->setGradientNode(nodeName, bsz, gData);
    }

    /// get gradient node (relevant for additional outputs)
    /// @param[in] skyNet - object net
    /// @param[in] nodeName - name node
    /// @param[out] gsz - data size
    /// @param[out] gData - data. First pass NULL, then pass it to the same 
    /// @return true - ok
    bool snGetGradientNode(skyNet fn,
        const char* nodeName,
        snLSize* gsz,
        snFloat** gData){

        if (!fn) return false;

        if (!nodeName || !gsz){
            static_cast<SNet*>(fn)->statusMess("SN error: !nodeName || !gsz");
            return false;
        }

        SN_Base::snSize bsz;
        if (!static_cast<SNet*>(fn)->getGradientNode(nodeName, bsz, gData)) return false;

        gsz->w = bsz.w;
        gsz->h = bsz.h;
        gsz->ch = bsz.d;
        gsz->bsz = bsz.n;

        return true;
    }

    /// set params of node
    /// @param[in] skyNet - object net
    /// @param[in] nodeName - name node
    /// @param[in] jnParam - params of node in JSON. 
    /// @return true - ok
    bool snSetParamNode(skyNet fn, const char* nodeName, const char* jnParam){
        
        if (!fn) return false;

        if (!nodeName || !jnParam){
            static_cast<SNet*>(fn)->statusMess("SN error: !nodeName || !jnParam");
            return false;
        }
                
        return static_cast<SNet*>(fn)->setParamNode(nodeName, jnParam);
    }

    /// get params of node
    /// @param[in] skyNet - object net
    /// @param[in] nodeName - name node
    /// @param[out] jnParam - params of node in JSON
    /// @return true - ok
    bool snGetParamNode(skyNet fn, const char* nodeName, char** jnParam){

        if (!fn) return false;

        if (!nodeName){
            static_cast<SNet*>(fn)->statusMess("SN error: !nodeName || !jnParam");
            return false;
        }

        return static_cast<SNet*>(fn)->getParamNode(nodeName, jnParam);
    }

    /// get architecture of net
    /// @param[in] skyNet - object net
    /// @param[out] jnNet - architecture of net in JSON
    /// @return true - ok
    bool snGetArchitecNet(skyNet fn, char** jnArchitecNet){

        if (!fn) return false;

        if (!jnArchitecNet){
            static_cast<SNet*>(fn)->statusMess("SN error: !jnArchitecNet");
            return false;
        }

        return static_cast<SNet*>(fn)->getArchitecNet(jnArchitecNet);
    }

    /// add user callBack for 'userLayer' node
    /// @param[in] skyNet - object net
    /// @param[in] cbackName - name 
    /// @param[in] snUserCBack - callBack
    /// @return true - ok
    bool snAddUserCallBack(skyNet fn, const char* cbackName, snUserCBack ucb, snUData ud){

        if (!fn) return false;

        if (!cbackName){
            static_cast<SNet*>(fn)->statusMess("SN error: !cbackName");
            return false;
        }

        return static_cast<SNet*>(fn)->snAddUserCallBack(cbackName, ucb, ud);
    }

    /// save all weight's (and bnorm if exist) to file
    /// @param[in] skyNet - object net
    /// @param[in] filePath - path to file
    /// @return true - ok
    bool snSaveAllWeightToFile(skyNet fn, const char* filePath){

        if (!fn) return false;

        if (!filePath){
            static_cast<SNet*>(fn)->statusMess("SN error: !filePath");
            return false;
        }

        return static_cast<SNet*>(fn)->saveAllWeightToFile(filePath);
    }

    /// load all weight's (and bnorm if exist) from file
    /// @param[in] skyNet - object net
    /// @param[in] filePath - path to file
    /// @return true - ok
    bool snLoadAllWeightFromFile(skyNet fn, const char* filePath){

        if (!fn) return false;

        if (!filePath){
            static_cast<SNet*>(fn)->statusMess("SN error: !filePath");
            return false;
        }

        return static_cast<SNet*>(fn)->loadAllWeightFromFile(filePath);
    }

    /// free object net
    /// @param[in] skyNet - object net
    void snFreeNet(skyNet fn){

        if (fn) delete static_cast<SNet*>(fn);
    }

    /// free resources
    /// @param[in] data - gradient or weight from snGetWeightNode
    /// @param[in] str - string from snGetArchitecNet
    void snFreeResources(snFloat* data, char* str){

        if (data) free(data);
        if (str) free(str);
    }
}
