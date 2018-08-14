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

#include <algorithm>
#include <iterator>
#include "stdafx.h"
#include "snBase/snBase.h"
#include "snet.h"


using namespace std;
using namespace SN_Base;

void SNet::statusMess(const string& mess){

	if (stsCBack_) stsCBack_(mess.c_str(), udata_);
}

// проверка перекр ссылок мду узлами
bool SNet::checkCrossRef(std::map<std::string, SN_Base::Node>& nodes, std::string& err){
		
	for (auto& n : nodes){

		// проверка наличия узлов
		if ((n.second.name == "") || (n.second.oprName == "")){
			err = "Error createNet: node '" + n.first + "' - not found";
			return false;
		}

		for (auto& next : n.second.nextNodes){

			if (nodes.find(next) == nodes.end()){
				err = "Error createNet: node '" + n.first + "' - not found next node '" + next + "'";
				statusMess(err);
				return false;
			}
		}

		for (auto& prev : n.second.prevNodes){

			if (nodes.find(prev) == nodes.end()){
				err = "Error createNet: node '" + n.first + "' - not found prev node '" + prev + "'";
				statusMess(err);
				return false;
			}
		}

		for (auto& next : n.second.nextNodes){

			auto& prevNodes = nodes[next].prevNodes;

			if (find(prevNodes.begin(), prevNodes.end(), n.first) == prevNodes.end()){

				err = "Error createNet: node '" + next + "' - not found prevNode '" + n.first + "'";
				statusMess(err);
				return false;
			}
		}

		for (auto& prev : n.second.prevNodes){

			auto& nextNodes = nodes[prev].nextNodes;

			if (find(nextNodes.begin(), nextNodes.end(), n.first) == nextNodes.end()){
				err = "Error createNet: node '" + prev + "' - not found nextNode '" + n.first + "'";
				statusMess(err);
				return false;
			}
		}
	}

	return true;
}

// создание сети
bool SNet::createNet(Net& inout_net, std::string& out_err){
		
	// проверка перекр ссылок мду узлами
	if (!checkCrossRef(inout_net.nodes, out_err)) return false;

	for (auto& n : inout_net.nodes){

		OperatorBase* opr = SN_Opr::createOperator(n.second.oprName, n.first, n.second.oprPrms);

		if (!opr){
			out_err = "Error createNet: not found operator '" + n.second.oprName + "'";
			inout_net.operats.clear();
			return false;
		}
		inout_net.operats[n.first] = opr;
	}
		
	for (auto& opr : inout_net.operats){
		weight_[opr.first] = new SN_Base::Tensor();
		opr.second->setWeight(weight_[opr.first]);

		inData_[opr.first] = new SN_Base::Tensor();
		opr.second->setInput(inData_[opr.first]);
		
		gradData_[opr.first] = new SN_Base::Tensor();
		opr.second->setGradient(gradData_[opr.first]);
	}
			
	return true;
}

/// создать нсеть
SNet::SNet(const char* jnNet, char* out_err /*sz 256*/,
	SN_API::snStatusCBack sts, SN_API::snUData ud) : stsCBack_(sts), udata_(ud){

	string err;  SN_Base::Net net;
	if (!jnParseNet(jnNet, net, err)){
		statusMess(err);
		strcpy(out_err, err.c_str());
		return;
	}
	
	SN_Opr::setStatusCBack(sts, ud);

	if (!createNet(net, err)){
		statusMess(err);
		strcpy(out_err, err.c_str());
		return;
	}

	nodes_ = net.nodes;
	operats_ = net.operats;

	engine_ = new SN_Eng::SNEngine(net, sts, ud);
}

SNet::~SNet(){

	if (engine_) delete engine_;

	for (auto o : operats_)
		SN_Opr::freeOperator(o.second, o.first);
}

/// тренинг
bool SNet::training(snFloat lr, snFloat* iLayer, const snSize& lsz, snFloat* targetData, snFloat* outData, const snSize& tsz, snFloat* outAccurate){
	
	// идем вперед
	if (!forward(true, iLayer, lsz, outData, tsz))
		return false;

	std::unique_lock<std::mutex> lk(mtxCmn_);
		
	// идем обратно	
	gradData_["EndNet"]->setData(targetData, tsz);

	lernParam_.lr = lr;
	lernParam_.action = snAction::backward;
	lernParam_.isAutoCalcError = true;
	lernParam_.isLerning = true;
	engine_->backward(lernParam_);

	// метрика
	auto outTensor = operats_["EndNet"]->getOutput();
	*outAccurate = calcAccurate(gradData_["EndNet"], outTensor);

	return true;
}

/// прямой проход
bool SNet::forward(bool isLern, snFloat* iLayer, const snSize& lsz, snFloat* outData, const snSize& osz){
    std::unique_lock<std::mutex> lk(mtxCmn_);
	
	if (!engine_){
		statusMess("forward error: net not create");
		return false;
	}
		
	if (((snSize)lsz).size() <= 0){
		statusMess("forward error: lsz.size() <= 0");
		return false;
	}

	inData_["BeginNet"]->setData(iLayer, lsz);

	lernParam_.action = snAction::forward;
	lernParam_.isLerning = isLern;
	engine_->forward(lernParam_);

	Tensor* tnsOut = operats_["EndNet"]->getOutput();

	auto& tnsOutSz = tnsOut->size();
	if (tnsOutSz != osz){
		statusMess("forward error: tnsOutSz != osz. Must be osz: " +
			to_string(tnsOutSz.w) + " " + to_string(tnsOutSz.h));
		return false;
	}

	memcpy(outData, tnsOut->getData(), tnsOutSz.size() * sizeof(snFloat));
		
	return true;
}

/// обратный проход
bool SNet::backward(snFloat lr, snFloat* gradErr, const snSize& gsz){
	std::unique_lock<std::mutex> lk(mtxCmn_);

	if (engine_){

		auto outTensor = operats_["EndNet"]->getOutput();
		
		auto& tsz = outTensor->size();
		if (tsz != gsz){
			statusMess("forward error: tnsOutSz != gsz. Must be gsz: " +
				to_string(tsz.w) + " " + to_string(tsz.h));
			return false;
		}
		
		gradData_["EndNet"]->setData(gradErr, outTensor->size());

		lernParam_.lr = lr;
		lernParam_.action = snAction::backward;
		lernParam_.isAutoCalcError = false;
		lernParam_.isLerning = true;
		engine_->backward(lernParam_);
	}
	else{
		statusMess("backward error: net not create");
		return false;
	}

	return true;
}

// считаем метрику
SN_Base::snFloat SNet::calcAccurate(Tensor* targetTens, Tensor* outTens){

	snFloat* targetData = targetTens->getData();
	snFloat* outData = outTens->getData();
	
	size_t accCnt = 0, bsz = outTens->size().n, osz = outTens->size().w;
	for (int i = 0; i < bsz; ++i){

		float* refTarget = targetData + i * osz;
		float* refOutput = outData + i * osz;

		if (osz > 1){
			int maxOutInx = distance(refOutput, max_element(refOutput, refOutput + osz)),
				maxTargInx = distance(refTarget, max_element(refTarget, refTarget + osz));

			if (maxTargInx == maxOutInx)
				++accCnt;
		}
		else{

			if (abs(refOutput[0] - refTarget[0]) < 0.1)
				++accCnt;
		}		
	}

	return (accCnt * 1.F) / bsz;
}

/// задать веса узла сети
bool SNet::setWeightNode(const char* nodeName, const SN_Base::snFloat* inData, const SN_Base::snSize& dsz){
	std::unique_lock<std::mutex> lk(mtxCmn_);

	if (operats_.find(nodeName) == operats_.end()) return false;
		
    weight_[nodeName]->setData((SN_Base::snFloat*)inData, dsz);
			
	return true;
}

/// вернуть веса узла сети
bool SNet::getWeightNode(const char* nodeName, SN_Base::snFloat** outData, SN_Base::snSize& dsz){
	std::unique_lock<std::mutex> lk(mtxCmn_);

	if (operats_.find(nodeName) == operats_.end()) return false;
	
	dsz = weight_[nodeName]->size();

	*outData = (snFloat*)realloc(*outData, dsz.size() * sizeof(snFloat));
		
	memcpy(*outData, weight_[nodeName]->getData(), dsz.size() * sizeof(snFloat));

	return true;
}

/// задать нормализацию для узла
bool SNet::setBatchNormNode(const char* nodeName, const SN_Base::batchNorm& bn){
	std::unique_lock<std::mutex> lk(mtxCmn_);

	if (operats_.find(nodeName) == operats_.end()) return false;

	snSize tsz = operats_[nodeName]->getOutput()->size();

	if ((tsz.w != bn.sz.w) || (tsz.h != bn.sz.h) || (tsz.d != bn.sz.d) || (bn.sz.n > 1)){
		statusMess("setBatchNormNode error: tsz != dsz. Must be dsz: " +
			to_string(tsz.w) + " " + to_string(tsz.h) + " " + to_string(tsz.d));
		return false;
	}

	operats_[nodeName]->setBatchNorm(bn);

	return true;
}

/// вернуть нормализацию узла
bool SNet::getBatchNormNode(const char* nodeName, SN_Base::batchNorm& obn){
	std::unique_lock<std::mutex> lk(mtxCmn_);

	if (operats_.find(nodeName) == operats_.end()) return false;

	obn = operats_[nodeName]->getBatchNorm();

	return true;
}

/// задать входные данные узла (актуально для доп входов)
bool SNet::setInputNode(const char* nodeName, const SN_Base::snFloat* inData, const SN_Base::snSize& dsz){
	std::unique_lock<std::mutex> lk(mtxCmn_);

	if (operats_.find(nodeName) == operats_.end()) return false;

	inData_[nodeName]->setData((SN_Base::snFloat*)inData, dsz);

	return true;
}

/// вернуть выходные значения узла (актуально для доп выходов)
bool SNet::getOutputNode(const char* nodeName, SN_Base::snFloat** outData, SN_Base::snSize& outSz){
	std::unique_lock<std::mutex> lk(mtxCmn_);

	if (operats_.find(nodeName) == operats_.end()) return false;

	Tensor* outTns = operats_[nodeName]->getOutput();

	outSz = outTns->size();

	*outData = (snFloat*)realloc(*outData, outSz.size() * sizeof(snFloat));

	memcpy(*outData, outTns->getData(), outSz.size() * sizeof(snFloat));

	return true;
}

/// задать градиент значения узла (актуально для доп выходов)
bool SNet::setGradientNode(const char* nodeName, const SN_Base::snFloat* inData, const SN_Base::snSize& dsz){
	std::unique_lock<std::mutex> lk(mtxCmn_);

	if (operats_.find(nodeName) == operats_.end()) return false;

	snSize tsz = gradData_[nodeName]->size();

	if (tsz != dsz){
		statusMess("setGradientNode error: tsz != dsz. Must be dsz: " +
			to_string(tsz.w) + " " + to_string(tsz.h) + " " + to_string(tsz.d));
		return false;
	}

	gradData_[nodeName]->setData((SN_Base::snFloat*)inData, dsz);

	return true;
}

/// вернуть градиент значения узла (актуально для доп выходов)
bool SNet::getGradientNode(const char* nodeName, SN_Base::snFloat** outData, SN_Base::snSize& outSz){
	std::unique_lock<std::mutex> lk(mtxCmn_);
	
	if (operats_.find(nodeName) == operats_.end()) return false;

	outSz = gradData_[nodeName]->size();

	*outData = (snFloat*)realloc(*outData, outSz.size() * sizeof(snFloat));

	memcpy(*outData, gradData_[nodeName]->getData(), outSz.size() * sizeof(snFloat));

	return true;
}
