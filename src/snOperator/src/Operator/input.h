
#pragma once

#include "snBase/snBase.h"


/// начало сети - оператор заглушка - ничего не должен делать!
class Input : SN_Base::OperatorBase{

public:

	Input(const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

	~Input() = default;

	/// задать аргументы для расчета
	bool setInput(SN_Base::Tensor* args) override;
			
	std::vector<std::string> Do(const SN_Base::learningParam&, const std::vector<OperatorBase*>& neighbOpr) override;
};