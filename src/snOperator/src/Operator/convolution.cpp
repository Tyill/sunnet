
#include "../stdafx.h"
#include "convolution.h"
#include "snAux/auxFunc.h"
#include "SNOperator/src/weightInit.h"
#include "SNOperator/src/activeFunctions.h"
#include "SNOperator/src/optimizer.h"
#include "SNOperator/src/structurs.h"
#include "SNOperator/src/mathFunctions.h"

using namespace std;
using namespace SN_Base;

/// сверточный слой

Convolution::Convolution(const string& name, const string& node, std::map<std::string, std::string>& prms) :
	OperatorBase(name, node, prms){
		
	load(prms);
}

void Convolution::load(std::map<std::string, std::string>& prms){

	baseOut_ = new Tensor();
	baseGrad_ = new Tensor();
	baseWeight_ = new Tensor();	

	auto setIntParam = [&prms](const string& name, bool isZero, size_t& value){

		if ((prms.find(name) != prms.end()) && SN_Aux::is_number(prms[name])){

			size_t v = stoi(prms[name]);
			if ((v > 0) || (isZero && (v == 0)))
				value = v;
			else
				statusMess("Convolution::setInternPrm error: param '" + name + (isZero ? "' < 0" : "' <= 0"));
		}
		else
			statusMess("Convolution::setInternPrm error: not found (or not numder) param '" + name + "'");
	};

	setIntParam("kernel", false, kernel_);
	setIntParam("krnWidth", false, krnWidth_);
	setIntParam("krnHeight", false, krnHeight_);
	setIntParam("padding", true, padding_);
	setIntParam("stride", true, stride_);
			
	// вспом массивы
	auxParams_["dWeight"] = vector<snFloat>();
	auxParams_["dWPrev"] = vector<snFloat>();
	auxParams_["dWGrad"] = vector<snFloat>();
	auxParams_["bn_norm"] = vector<snFloat>();               bnPrm_.norm = auxParams_["bn_norm"].data();
	auxParams_["bn_mean"] = vector<snFloat>(kernel_, 0);     bnPrm_.mean = auxParams_["bn_mean"].data();
	auxParams_["bn_varce"] = vector<snFloat>(kernel_, 0);    bnPrm_.varce = auxParams_["bn_varce"].data();
	auxParams_["bn_scale"] = vector<snFloat>(kernel_, 1.F);  bnPrm_.scale = auxParams_["bn_scale"].data();
	auxParams_["bn_dScale"] = vector<snFloat>(kernel_, 0);   bnPrm_.dScale = auxParams_["bn_dScale"].data();
	auxParams_["bn_schift"] = vector<snFloat>(kernel_, 0);   bnPrm_.schift = auxParams_["bn_schift"].data();
	auxParams_["bn_dSchift"] = vector<snFloat>(kernel_, 0);  bnPrm_.dSchift = auxParams_["bn_dSchift"].data();
	auxParams_["bn_onc"] = vector<snFloat>();	             bnPrm_.onc = auxParams_["bn_onc"].data();

	setInternPrm(prms);
}

bool Convolution::setInternPrm(std::map<std::string, std::string>& prms){

	basePrms_ = prms;

	if (prms.find("activeType") != prms.end()){

		string atype = prms["activeType"];
		if (atype == "none") activeType_ = activeType::none;
		else if (atype == "sigmoid") activeType_ = activeType::sigmoid;
		else if (atype == "relu") activeType_ = activeType::relu;
		else if (atype == "leakyRelu") activeType_ = activeType::leakyRelu;
		else if (atype == "elu") activeType_ = activeType::elu;
		else
			statusMess("FullyConnected::setInternPrm error: param 'activeType' = " + atype + " indefined");
	}

	if (prms.find("optimizerType") != prms.end()){

		string optType = prms["optimizerType"];
		if (optType == "sgd") optimizerType_ = optimizerType::sgd;
		else if (optType == "sgdMoment") optimizerType_ = optimizerType::sgdMoment;
		else if (optType == "adagrad") optimizerType_ = optimizerType::adagrad;
		else if (optType == "adam") optimizerType_ = optimizerType::adam;
		else if (optType == "RMSprop") optimizerType_ = optimizerType::RMSprop;
		else
			statusMess("FullyConnected::setInternPrm error: param 'optimizerType' = " + optType + " indefined");
	}

	if (prms.find("weightInitType") != prms.end()){

		string wInit = prms["weightInitType"];
		if (wInit == "uniform") weightInitType_ = weightInitType::uniform;
		else if (wInit == "he") weightInitType_ = weightInitType::he;
		else if (wInit == "lecun") weightInitType_ = weightInitType::lecun;
		else if (wInit == "xavier") weightInitType_ = weightInitType::xavier;
		else
			statusMess("FullyConnected::setInternPrm error: param 'weightInitType' = " + wInit + " indefined");
	}

	if (prms.find("batchNormType") != prms.end()){

		string bnType = prms["batchNormType"];
		if (bnType == "none") batchNormType_ = batchNormType::none;
		else if (bnType == "beforeActive") batchNormType_ = batchNormType::beforeActive;
		else if (bnType == "postActive") batchNormType_ = batchNormType::postActive;
		else
			statusMess("FullyConnected::setInternPrm error: param 'batchNormType' = " + bnType + " indefined");
	}

	if (prms.find("decayMomentDW") != prms.end())
		opt_decayMomentDW_ = stof(prms["decayMomentDW"]);

	if (prms.find("decayMomentWGr") != prms.end())
		opt_decayMomentWGr_ = stof(prms["decayMomentWGr"]);

	if (prms.find("lmbRegular") != prms.end())
		opt_lmbRegular_ = stof(prms["lmbRegular"]);

	if (prms.find("batchNormLr") != prms.end())
		bnPrm_.lr = stof(prms["batchNormLr"]);
				
	return true;
}

/// выполнить расчет
std::vector<std::string> Convolution::Do(const learningParam& lernPrm, const std::vector<OperatorBase*>& neighbOpr){
		
	if (neighbOpr.size() == 1){
		if (lernPrm.action == snAction::forward)
			forward(neighbOpr[0]->getOutput());
		else
			backward(neighbOpr[0]->getGradient(), lernPrm);
	}
	else{
		if (lernPrm.action == snAction::forward){
		
			inFwTns_ = *neighbOpr[0]->getOutput();

			int sz = neighbOpr.size();
			for (size_t i = 1; i < sz; ++i)
				inFwTns_ += *neighbOpr[i]->getOutput();
			
			forward(&inFwTns_);
		}
		else{
		
			inBwTns_ = *neighbOpr[0]->getGradient();

			int sz = neighbOpr.size();
			for (size_t i = 1; i < sz; ++i)
				inBwTns_ += *neighbOpr[i]->getGradient();
						
			backward(&inBwTns_, lernPrm);
		}
	}

	return std::vector<std::string>();
}

void Convolution::forward(SN_Base::Tensor* inTns){

	snSize insz = inTns->size();

	/// размер вх данных изменился?
	if (insz != inSzMem_){
		inSzMem_ = insz;
		updateConfig(insz);
	}

	/// копируем со смещением padding для каждого изобр
	snFloat* pInTns = inTns->getData();
	snFloat* pDtMem = inDataExp_.data();

	if (padding_ == 0)
		memcpy(pDtMem, pInTns, insz.size() * sizeof(snFloat));
	else{
		size_t padding = padding_, sz = insz.h * insz.d * insz.n, stW = insz.w, stH = insz.h;
		for (size_t i = 0; i < sz; ++i){

			if (i % stH == 0)
				pDtMem += (stW + padding * 2) * padding;

			pDtMem += padding;
			for (size_t j = 0; j < stW; ++j)
				pDtMem[j] = pInTns[j];
			pDtMem += padding + stW;

			pInTns += stW;
		}
	}
	
	/// расчет выходных значений нейронов
	snFloat* out = baseOut_->getData(), *weight = baseWeight_->getData();
	snSize outsz = baseOut_->size();
	fwdConvolution(kernel_, krnWidth_, krnHeight_, stride_, weight, inDataExpSz_, inDataExp_.data(), outsz, out);

	/// batchNorm
	if (batchNormType_ == batchNormType::beforeActive)
		fwdBatchNorm(outsz, out, out, bnPrm_);

	/// функция активации
	switch (activeType_){
	case activeType::sigmoid:   fv_sigmoid(out, outsz.size()); break;
	case activeType::relu:      fv_relu(out, outsz.size()); break;
	case activeType::leakyRelu: fv_leakyRelu(out, outsz.size()); break;
	case activeType::elu:       fv_elu(out, outsz.size()); break;
	default: break;
	}

	/// batchNorm
	if (batchNormType_ == batchNormType::postActive)
		fwdBatchNorm(outsz, out, out, bnPrm_);
}

void Convolution::backward(SN_Base::Tensor* inTns, const learningParam& lernPrm){

	snFloat* gradIn = inTns->getData();

	/// batchNorm
	if (batchNormType_ == batchNormType::postActive)
		bwdBatchNorm(inTns->size(), gradIn, gradIn, bnPrm_);

	// проходим через ф-ю активации, если есть
	if (activeType_ != activeType::none){

		snFloat* out = baseOut_->getData();
		
		// производная функции активации
		size_t ksz = kernel_ * inSzMem_.n;
		switch (activeType_){
		case activeType::sigmoid:   df_sigmoid(out, ksz); break;
		case activeType::relu:      df_relu(out, ksz); break;
		case activeType::leakyRelu: df_leakyRelu(out, ksz); break;
		case activeType::elu:       df_elu(out, ksz); break;
		default: break;
		}

		// обновл градиент
		for (size_t i = 0; i < ksz; ++i) gradIn[i] *= out[i];
	}

	/// batchNorm
	if (batchNormType_ == batchNormType::beforeActive)
		bwdBatchNorm(inTns->size(), gradIn, gradIn, bnPrm_);

	// расчет вых градиента и коррекции весов
	snFloat* gradOut = baseGrad_->getData();
	snFloat* weight = baseWeight_->getData();
	snFloat* dWeight = auxParams_["dWeight"].data();
	bwdConvolution(kernel_, krnWidth_, krnHeight_, stride_, weight, inDataExpSz_, inDataExp_.data(),
		baseOut_->size(), gradIn, gradOut, dWeight);
		
	// корректируем веса
	snFloat* dWPrev = auxParams_["dWPrev"].data();
	snFloat* dWGrad = auxParams_["dWGrad"].data();
	size_t wsz = baseWeight_->size().size();
	
	switch (optimizerType_){
	case optimizerType::sgd:       opt_sgd(dWeight, weight, wsz, lernPrm.lr, opt_lmbRegular_); break;
	case optimizerType::sgdMoment: opt_sgdMoment(dWeight, dWPrev, weight, wsz, lernPrm.lr, opt_lmbRegular_, opt_decayMomentDW_); break;
	case optimizerType::RMSprop:   opt_RMSprop(dWeight, dWGrad, weight, wsz, lernPrm.lr, opt_lmbRegular_, opt_decayMomentWGr_); break;
	case optimizerType::adagrad:   opt_adagrad(dWeight, dWGrad, weight, wsz, lernPrm.lr, opt_lmbRegular_); break;
	case optimizerType::adam:      opt_adam(dWeight, dWPrev, dWGrad, weight, wsz, lernPrm.lr, opt_lmbRegular_, opt_decayMomentDW_, opt_decayMomentWGr_); break;
	default: break;
	}

}

void Convolution::updateConfig(const snSize& newsz){
	
	size_t stp = krnWidth_ * krnHeight_ * newsz.d, ntp = (stp + 1) * kernel_;
		
	// имеющиеся веса оставляем как есть, остаток инициализируем
	size_t wcsz = baseWeight_->size().size();
	if (ntp > wcsz){
				
		baseWeight_->resize(snSize(kernel_, stp + 1));
		snFloat* wd = baseWeight_->getData();
		switch (weightInitType_){
		case weightInitType::uniform: wi_uniform(wd + wcsz, ntp - wcsz); break;
		case weightInitType::he: wi_he(wd + wcsz, ntp - wcsz, stp + 1); break;
		case weightInitType::lecun:wi_lecun(wd + wcsz, ntp - wcsz, kernel_); break;
		case weightInitType::xavier:wi_xavier(wd + wcsz, ntp - wcsz, stp + 1, kernel_); break;
		}
	}

	inDataExpSz_ = snSize(newsz.w + padding_ * 2, newsz.h + padding_ * 2, newsz.d, newsz.n);
	inDataExp_.resize(inDataExpSz_.size());

	memset(inDataExp_.data(), 0, inDataExpSz_.size() * sizeof(snFloat));
	

	snSize outSz(0, 0, kernel_, newsz.n, 1);
	for (int i = krnWidth_ / 2; i < (newsz.w + padding_ * 2 - krnWidth_ / 2); i += stride_)
		++outSz.w;

	for (int i = krnHeight_ / 2; i < (newsz.h + padding_ * 2 - krnHeight_ / 2); i += stride_)
		++outSz.h;

	baseOut_->resize(outSz);
	baseGrad_->resize(outSz);
		
	// вспом массивы
	auxParams_["dWeight"].resize(ntp, 0);
	auxParams_["dWPrev"].resize(ntp, 0);
	auxParams_["dWGrad"].resize(ntp, 0);
	auxParams_["bn_norm"].resize(newsz.n * kernel_, 0);   bnPrm_.norm = auxParams_["bn_norm"].data();
	auxParams_["bn_onc"].resize(newsz.n, 1.F);            bnPrm_.onc = auxParams_["bn_onc"].data();
} 



