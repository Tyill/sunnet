
#pragma once

#include "snBase/snBase.h"
#include"SNOperator/src/structurs.h"
#include"SNOperator/src/mathFunctions.h"

/// прямой проход
void fwdFullyConnected(size_t kernel,   ///< размер скрыт слоя
	            SN_Base::snSize insz,   ///< вход значения размер 
		     SN_Base::snFloat* input,   ///< вход значения
			SN_Base::snFloat* weight,   ///< веса
		    SN_Base::snFloat* output);  ///< выход знач (скрытых нейронов) для след слоя

/// обратный проход
void bwdFullyConnected(size_t kernel,   ///< размер скрыт слоя
	        SN_Base::snFloat* weight,   ///< веса
                SN_Base::snSize insz,   ///< вход значения размер 
	         SN_Base::snFloat* input,   ///< вход значения 
	        SN_Base::snFloat* gradIn,   ///< вход градиент ошибки с пред слоя
	       SN_Base::snFloat* gradOut,   ///< выход градиент ошибки для след слоя
	    SN_Base::snFloat* dWeightOut);  ///< дельта изменения весов


/// полносвязный слой
class FullyConnected : SN_Base::OperatorBase{

public:

	FullyConnected(const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

	~FullyConnected() = default;

	std::vector<std::string> Do(const SN_Base::learningParam&, const std::vector<OperatorBase*>& neighbOpr) override;
		
	bool setInternPrm(std::map<std::string, std::string>& prms);
		
private:
		
	size_t kernel_ = 10;                                        ///< кол-во скрытых нейронов
														        
	activeType activeType_ = activeType::none;                  ///< тип ф-ии активации
	optimizerType optimizerType_ = optimizerType::sgd;          ///< тип оптимизатора весов
	weightInitType weightInitType_ = weightInitType::uniform;   ///< тип инициализации весов
	batchNormType batchNormType_ = batchNormType::none;         ///< тип batchNorm 
	SN_Base::snSize inSzMem_;                                   ///< размер вх данных
	std::vector<SN_Base::snFloat> inDataExp_;                   ///< вход данные расширен

	batchNormParam bnPrm_;                                      ///< параметры batchNorm

	SN_Base::Tensor inFwTns_, inBwTns_;                         ///< тензор с сосед слоя 

	SN_Base::snFloat opt_decayMomentDW_ = 0.9F,                 ///< оптимизация изм весов
		             opt_decayMomentWGr_ = 0.99F,
		             opt_lmbRegular_ = 0.001F;

	std::map<std::string, std::vector<SN_Base::snFloat>> auxParams_;  ///< вспом данные для расчета
		
	void load(std::map<std::string, std::string>& prms);

	void updateConfig(const SN_Base::snSize& newSz);
		
	void forward(SN_Base::Tensor* inTns);
	void backward(SN_Base::Tensor* inTns, const SN_Base::learningParam& lernPrm);

};