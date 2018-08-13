
#pragma once

#include "snBase/snBase.h"


/// конец сети - оператор заглушка - ничего не должен делать!
class Output : SN_Base::OperatorBase{

public:

	Output(const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

	~Output() = default;

	std::vector<std::string> Do(const SN_Base::learningParam&, const std::vector<OperatorBase*>& neighbOpr) override;

private:

	SN_Base::Tensor inFwTns_;                    ///< тензор с сосед слоя 

};