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

		return new SNet(jnNet, out_err, sts, ud);
	}
	

	/// training - a cycle forward-back with auto-correction of weights
	/// @param[in] skyNet - object net
	/// @param[in] lr - learning rate
	/// @param[in] iLayer - input layer
	/// @param[in] lsz - input layer size
	/// @param[in] targetData - target, the size must match the markup. The memory is allocated by the user
	/// @param[out] outData - result, the size must match the markup. The memory is allocated by the user
	/// @param[in] tsz - size of target and result. Sets for verification
	/// @param[out] outAccurate - current accuracy
	/// @return true - ok
	bool snTraining(skyNet fn,
		            snFloat lr,
		            snFloat* iLayer,
		            snLSize lsz,
		            snFloat* targetData,
		            snFloat* outData,
		            snLSize tsz,
		            snFloat* outAccurate){

		if (!fn) return false;

		SN_Base::snSize bsz(lsz.w, lsz.h, lsz.ch, lsz.bch);
		SN_Base::snSize tnsz(tsz.w, tsz.h, tsz.ch, tsz.bch);

		return static_cast<SNet*>(fn)->training(lr, iLayer, bsz, targetData, outData, tnsz, outAccurate);
	}

	/// forward pass
	/// @param[in] skyNet - object net
	/// @param[in] isLern - is lern?
	/// @param[in] iLayer - input layer
	/// @param[in] lsz - input layer size
	/// @param[out] outData - result, the size must match the markup. The memory is allocated by the user
	/// @param[in] osz - size of result. Sets for verification
	/// @return true - ok
	bool snForward(skyNet fn,
		           bool isLern,
		           snFloat* iLayer,
		           snLSize lsz,
		           snFloat* outData,
		           snLSize osz){

		if (!fn) return false;

		SN_Base::snSize bsz(lsz.w, lsz.h, lsz.ch, lsz.bch);
		SN_Base::snSize onsz(osz.w, osz.h, osz.ch, osz.bch);

		return static_cast<SNet*>(fn)->forward(isLern, iLayer, bsz, outData, onsz);
	}

	/// backward pass
	/// @param[in] skyNet - object net
	/// @param[in] lr - learning rate
	/// @param[in] inGradErr - error gradient, the size must match the output
	/// @param[in] gsz - size of the error gradient. Sets for verification
	/// @return true - ok
	bool snBackward(skyNet fn,
		            snFloat lr,
		            snFloat* inGradErr,
		            snLSize gsz){

		if (!fn) return false;

		SN_Base::snSize gnsz(gsz.w, gsz.h, gsz.ch, gsz.bch);

		return static_cast<SNet*>(fn)->backward(lr, inGradErr, gnsz);
	}

	/// set weight of node
	/// @param[in] skyNet - object net
	/// @param[in] nodeName - name node
	/// @param[in] inData - data
	/// @param[in] dsz - data size
	/// @return true - ok
	bool snSetWeightNode(skyNet fn, const char* nodeName, const snFloat* inData, snLSize dsz){

		if (!fn) return false;

		SN_Base::snSize bsz(dsz.w, dsz.h, dsz.ch, dsz.bch);

		return static_cast<SNet*>(fn)->setWeightNode(nodeName, inData, bsz);
	}

	/// get weight of node
	/// @param[in] skyNet - object net
	/// @param[in] nodeName - name node
	/// @param[out] outData - output data. First pass NULL, then pass it to the same 
	/// @param[out] outSz - output size
	/// @return true - ok
	bool snGetWeightNode(skyNet fn, const char* nodeName, snFloat** outData, snLSize* dsz){

		if (!fn) return false;

		SN_Base::snSize bsz;
		if (!static_cast<SNet*>(fn)->getWeightNode(nodeName, outData, bsz)) return false;

		dsz->w = bsz.w;
		dsz->h = bsz.h;
		dsz->ch = bsz.d;
		dsz->bch = bsz.n;
				
		return true;
	}

	/// set batchNorm of node
	/// @param[in] skyNet - object net
	/// @param[in] nodeName - name node
	/// @param[in] inData - data
	/// @param[in] dsz - data size
	/// @return true - ok
	bool snSetBatchNormNode(skyNet fn, const char* nodeName, const SN_API::batchNorm inData, snLSize dsz){

		if (!fn) return false;

		SN_Base::batchNorm bn;
		SN_Base::snSize sz(dsz.w * dsz.h * dsz.ch);
		bn.set(inData.mean, inData.varce, inData.scale, inData.schift, sz);
		
		return static_cast<SNet*>(fn)->setBatchNormNode(nodeName, bn);
	}

	/// get batchNorm of node
	/// @param[in] skyNet - object net
	/// @param[in] nodeName - name node
	/// @param[out] outData - data 
	/// @param[out] outSz - data size
	/// @return true - ok
	bool snGetBatchNormNode(skyNet fn, const char* nodeName, batchNorm* outData, snLSize* outSz){

		if (!fn) return false;

		SN_Base::batchNorm bn;
		if (!static_cast<SNet*>(fn)->getBatchNormNode(nodeName, bn)) return false;

		size_t sz = bn.sz.size();

		outData->mean =   new snFloat[sz]; memcpy(outData->mean, bn.mean.data(),    sz * sizeof(snFloat));
		outData->varce =  new snFloat[sz]; memcpy(outData->varce, bn.varce.data(),  sz * sizeof(snFloat));
		outData->scale =  new snFloat[sz]; memcpy(outData->scale, bn.scale.data(),  sz * sizeof(snFloat));
		outData->schift = new snFloat[sz]; memcpy(outData->schift, bn.schift.data(),sz * sizeof(snFloat));

		*outSz = snLSize(bn.sz.w, bn.sz.h, bn.sz.d);

		return true;
	}
	
	/// set input node (relevant for additional inputs)
	/// @param[in] skyNet - object net
	/// @param[in] nodeName - name node
	/// @param[in] inData - data
	/// @param[in] dsz - data size
	/// @return true - ok
	bool snSetInputNode(skyNet fn,
		const char* nodeName,
		const snFloat* inData,
		snLSize dsz){

		if (!fn) return false;

		SN_Base::snSize bsz(dsz.w, dsz.h, dsz.ch, dsz.bch);

		return static_cast<SNet*>(fn)->setInputNode(nodeName, inData, bsz);
	}

	/// get output node (relevant for additional inputs)
	/// @param[in] skyNet - object net
	/// @param[in] nodeName - name node
	/// @param[out] outData - data. First pass NULL, then pass it to the same 
	/// @param[out] outSz - data size
	/// @return true - ok
	bool snGetOutputNode(skyNet fn,
		const char* nodeName,
		snFloat** outData,
		snLSize* dsz){

		if (!fn) return false;

		SN_Base::snSize bsz;
		if (!static_cast<SNet*>(fn)->getOutputNode(nodeName, outData, bsz)) return false;

		dsz->w = bsz.w;
		dsz->h = bsz.h;
		dsz->ch = bsz.d;
		dsz->bch = bsz.n;

		return true;

	}

	/// set gradient node (relevant for additional outputs)
	/// @param[in] skyNet - object net
	/// @param[in] nodeName - name node
	/// @param[in] inData - data
	/// @param[in] dsz - data size
	/// @return true - ok
	bool snSetGradientNode(skyNet fn,
		const char* nodeName,
		const snFloat* inData,
		snLSize dsz){

		if (!fn) return false;

		SN_Base::snSize bsz(dsz.w, dsz.h, dsz.ch, dsz.bch);

		return static_cast<SNet*>(fn)->setGradientNode(nodeName, inData, bsz);
	}

	/// get gradient node (relevant for additional outputs)
	/// @param[in] skyNet - object net
	/// @param[in] nodeName - name node
	/// @param[out] outData - data. First pass NULL, then pass it to the same 
	/// @param[out] outSz - data size
	/// @return true - ok
	bool snGetGradientNode(skyNet fn,
		const char* nodeName,
		snFloat** outData,
		snLSize* dsz){

		if (!fn) return false;

		SN_Base::snSize bsz;
		if (!static_cast<SNet*>(fn)->getGradientNode(nodeName, outData, bsz)) return false;

		dsz->w = bsz.w;
		dsz->h = bsz.h;
		dsz->ch = bsz.d;
		dsz->bch = bsz.n;

		return true;
	}

	/// set params of node
	/// @param[in] skyNet - object net
	/// @param[in] nodeName - name node
	/// @param[in] jnParam - params of node in JSON. 
	/// @return true - ok
	bool snSetParamNode(skyNet fn, const char* nodeName, const char* jnParam){
		
		if (!fn) return false;
				
		return static_cast<SNet*>(fn)->setParamNode(nodeName, jnParam);
	}

	/// get params of node
	/// @param[in] skyNet - object net
	/// @param[in] nodeName - name node
	/// @param[out] jnParam - params of node in JSON. The memory is allocated by the user 
	/// @return true - ok
	bool snGetParamNode(skyNet fn, const char* nodeName, char* jnParam /*minsz 256*/){

		if (!fn) return false;

		return static_cast<SNet*>(fn)->getParamNode(nodeName, jnParam);
	}

	/// get architecture of net
	/// @param[in] skyNet - object net
	/// @param[out] jnNet - architecture of net in JSON. The memory is allocated by the user
	/// @return true - ok
	bool snGetArchitecNet(skyNet fn, char* jnArchitecNet /*minsz 2048*/){

		if (!fn) return false;

		return static_cast<SNet*>(fn)->getArchitecNet(jnArchitecNet);
	}

	/// free object net
	/// @param[in] skyNet - object net
	void snFreeNet(skyNet fn){

		if (fn) delete static_cast<SNet*>(fn);
	}
}