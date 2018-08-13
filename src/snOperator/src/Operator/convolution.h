
#pragma once

#include "snBase/snBase.h"
#include"SNOperator/src/structurs.h"
#include"SNOperator/src/mathFunctions.h"

/// прямой проход
void fwdConvolution(size_t kernel,   ///< колво вых слоев
                  size_t krnWidth,   ///< ширина маски
				 size_t krnHeight,   ///< высота маски
				    size_t stride,   ///< шаг движения маски
         SN_Base::snFloat* weight,   ///< веса
		     SN_Base::snSize insz,   ///< вход значения размер 
	      SN_Base::snFloat* input,   ///< вход значения
		    SN_Base::snSize outsz,   ///< выход значения размер 
		 SN_Base::snFloat* output);  ///< выход знач (скрытых нейронов) для след слоя

/// обратный проход
void bwdConvolution(size_t kernel,   ///< колво вых слоев
	              size_t krnWidth,   ///< ширина маски
	             size_t krnHeight,   ///< высота маски
	                size_t stride,   ///< шаг движения маски
	     SN_Base::snFloat* weight,   ///< веса
             SN_Base::snSize insz,   ///< вход значения размер 
	      SN_Base::snFloat* input,   ///< вход значения 
		    SN_Base::snSize outsz,   ///< выход значения размер 
	     SN_Base::snFloat* gradIn,   ///< вход градиент ошибки с пред слоя
	    SN_Base::snFloat* gradOut,   ///< выход градиент ошибки для след слоя
	 SN_Base::snFloat* dWeightOut);  ///< дельта изменения весов


/// сверточный слой
class Convolution : SN_Base::OperatorBase{

public:

	Convolution(const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

	~Convolution() = default;

	std::vector<std::string> Do(const SN_Base::learningParam&, const std::vector<OperatorBase*>& neighbOpr) override;
		
	bool setInternPrm(std::map<std::string, std::string>& prms) override;
	
private:
		
	size_t kernel_ = 10;                                        ///< кол-во вых слоев свертки
	size_t krnWidth_ = 3;                                       ///< длина слоя свертки
	size_t krnHeight_ = 3;                                      ///< высота слоя свертки
	size_t padding_ = 0;                                        ///< доп отступ по краям для свертки
	size_t stride_ = 1;                                         ///< шаг перемещения свертки

	activeType activeType_ = activeType::none;                  ///< тип ф-ии активации
	optimizerType optimizerType_ = optimizerType::sgd;          ///< тип оптимизатора весов
	weightInitType weightInitType_ = weightInitType::uniform;   ///< тип инициализации весов
	batchNormType batchNormType_ = batchNormType::none;         ///< тип batchNorm 
	SN_Base::snSize inSzMem_;                                   ///< размер вх данных
	SN_Base::snSize inDataExpSz_;                               ///< размер вх данных
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