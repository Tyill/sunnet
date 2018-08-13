
#pragma once

#include "snBase/snBase.h"


/// оператор - расчет ошибки
class LossFunction : SN_Base::OperatorBase{

public:

	enum lossType{
		softMaxACrossEntropy = 0,
		binaryCrossEntropy = 1,
	};

	LossFunction(const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

	~LossFunction() = default;
			
	std::vector<std::string> Do(const SN_Base::learningParam&, const std::vector<OperatorBase*>& neighbOpr) override;

private:
	lossType lossType_ = lossType::softMaxACrossEntropy;

	SN_Base::Tensor inFwTns_, inBwTns_;                              ///< тензор с сосед слоя 

	std::map<std::string, std::vector<SN_Base::snFloat>> auxParams_; ///< вспом данные для расчета

	void load(std::map<std::string, std::string>& prms);

	void forward(SN_Base::Tensor* inTns);
	void backward(SN_Base::Tensor* inTns, const SN_Base::learningParam& lernPrm);
};