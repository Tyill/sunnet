// skyNet.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "snBase/snBase.h"
#include "skynet/skynet.h"
#include "snet.h"


namespace SN_API{

	/// создать объект нсети
	/// @param jnNet - архитектура сети в JSON
	/// @param out_err - ошибка разбора jsNet. "" - ok. Память выделяет польз-ль.
	/// @param statusCBack - callback состояния. Необязательно
	/// @param udata - польз данные (для callback состояния). Необязательно
	skyNet snCreateNet(const char* jnNet,
		               char* out_err /*sz 256*/,
					   snStatusCBack sts,
					   snUData ud){

		return new SNet(jnNet, out_err, sts, ud);
	}
	
	/// тренинг - цикл вперед-назад с автокоррекцией весов
	/// @param skyNet - объект нсети
	/// @param lr - скорость обучения
	/// @param iLayer - входной слой
	/// @param lsz - размер вх слоя
	/// @param targetData - целевой результат, размер должен соот-ть разметке. Память выделяет польз-ль.
	/// @param outData - выходной результат, размер соот-ет разметке. Память выделяет польз-ль.
	/// @param tsz - размер целевой и выходного результата. Задается для проверки.
	/// @param outAccurate - текущая точность
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

	/// прямой проход
	/// @param skyNet - объект нсети
	/// @param isLern - обучение?
	/// @param iLayer - входной слой
	/// @param lsz - размер вх слоя
	/// @param outData - выходной результат, размер соот-ет разметке. Память выделяет польз-ль.
	/// @param osz - размер выходного результата. Задается для проверки.
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

	/// обратный проход
	/// @param skyNet - объект нсети
	/// @param lr - скорость обучения
	/// @param inGradErr - градиент ошибки, размер должен соот-ть выходному результату До softMax.
	///  например, если на выходе сети softMax (insm -> SM -> outsm), то градиент ошибки дбыть - для Входа в softMax (dL/dinsm)
	/// @param gsz - размер градиента ошибки. Задается для проверки.
	/// @return true - ok
	bool snBackward(skyNet fn,
		            snFloat lr,
		            snFloat* inGradErr,
		            snLSize gsz){

		if (!fn) return false;

		SN_Base::snSize gnsz(gsz.w, gsz.h, gsz.ch, gsz.bch);

		return static_cast<SNet*>(fn)->backward(lr, inGradErr, gnsz);
	}

	/// задать веса узла сети
	/// @param skyNet - объект нсети
	/// @param nodeName - имя узла
	/// @param inData - данные
	/// @param dsz - размер данных
	/// @return true - ok
	bool snSetWeightNode(skyNet fn, const char* nodeName, const snFloat* inData, snLSize dsz){

		if (!fn) return false;

		SN_Base::snSize bsz(dsz.w, dsz.h, dsz.ch, dsz.bch);

		return static_cast<SNet*>(fn)->setWeightNode(nodeName, inData, bsz);
	}

	/// вернуть веса узла сети
	/// @param skyNet - объект нсети
	/// @param nodeName - имя узла
	/// @param outData - данные. Память выделяет пользователь
	/// @param dsz - размер данных
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

	/// задать нормализацию для узла
	/// @param[in] skyNet - объект нсети
	/// @param[in] nodeName - имя узла
	/// @param[in] inData - данные
	/// @param[in] dsz - размер данных
	/// @return true - ok
	bool snSetBatchNormNode(skyNet fn, const char* nodeName, const SN_API::batchNorm inData, snLSize dsz){

		if (!fn) return false;

		SN_Base::batchNorm bn;
		SN_Base::snSize sz(dsz.w * dsz.h * dsz.ch);
		bn.set(inData.mean, inData.varce, inData.scale, inData.schift, sz);
		
		return static_cast<SNet*>(fn)->setBatchNormNode(nodeName, bn);
	}

	/// вернуть нормализацию узла
	/// @param[in] skyNet - объект нсети
	/// @param[in] nodeName - имя узла
	/// @param[out] outData - данные 
	/// @param[out] outSz - размер данных
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

	/// задать входные данные узла (актуально для доп входов)
	/// @param skyNet - объект нсети
	/// @param nodeName - имя узла
	/// @param inData - данные
	/// @param dsz - размер данных
	/// @return true - ok
	bool snSetInputNode(skyNet fn,
		const char* nodeName,
		const snFloat* inData,
		snLSize dsz){

		if (!fn) return false;

		SN_Base::snSize bsz(dsz.w, dsz.h, dsz.ch, dsz.bch);

		return static_cast<SNet*>(fn)->setInputNode(nodeName, inData, bsz);
	}

	/// вернуть выходные значения узла (актуально для доп выходов)
	/// @param skyNet - объект нсети
	/// @param nodeName - имя узла
	/// @param outData - данные. Сначала передать NULL, потом передавать его же. 
	/// @param outSz - размер данных
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

	/// задать градиент значения узла (актуально для доп выходов)
	/// @param skyNet - объект нсети
	/// @param nodeName - имя узла
	/// @param inData - данные
	/// @param dsz - размер данных
	/// @return true - ok
	bool snSetGradientNode(skyNet fn,
		const char* nodeName,
		const snFloat* inData,
		snLSize dsz){

		if (!fn) return false;

		SN_Base::snSize bsz(dsz.w, dsz.h, dsz.ch, dsz.bch);

		return static_cast<SNet*>(fn)->setGradientNode(nodeName, inData, bsz);
	}

	/// вернуть градиент значения узла (актуально для доп выходов)
	/// @param skyNet - объект нсети
	/// @param nodeName - имя узла
	/// @param outData - данные. Сначала передать NULL, потом передавать его же. 
	/// @param outSz - размер данных
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

	/// задать параметры узла
	/// @param skyNet - объект нсети
	/// @param nodeName - имя узла
	/// @param jnParam - параметры узла в JSON. 
	/// @return true - ok
	bool snSetParamNode(skyNet fn, const char* nodeName, const char* jnParam){
		
		if (!fn) return false;
				
		return static_cast<SNet*>(fn)->setParamNode(nodeName, jnParam);
	}

	/// вернуть параметры узла
	/// @param skyNet - объект нсети
	/// @param nodeName - имя узла
	/// @param jnParam - параметры узла в JSON. Память выделяет пользователь. 
	/// @return true - ok
	bool snGetParamNode(skyNet fn, const char* nodeName, char* jnParam /*minsz 256*/){

		if (!fn) return false;

		return static_cast<SNet*>(fn)->getParamNode(nodeName, jnParam);
	}

	/// вернуть архитектуру сети
	/// @param skyNet - объект нсети
	/// @param jsArchitecNet - архитектура сети в JSON. Память выделяет пользователь.
	/// @return true - ok
	bool snGetArchitecNet(skyNet fn, char* jnArchitecNet /*minsz 2048*/){

		if (!fn) return false;

		return static_cast<SNet*>(fn)->getArchitecNet(jnArchitecNet);
	}

	/// освободить объект сети
	/// @param skyNet - объект нсети
	void snFreeNet(skyNet fn){

		if (fn) delete static_cast<SNet*>(fn);
	}
}