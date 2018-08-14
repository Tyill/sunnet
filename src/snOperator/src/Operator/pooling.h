
#pragma once

#include "snBase/snBase.h"
#include"SNOperator/src/structurs.h"
#include"SNOperator/src/mathFunctions.h"


/// объединяющий слой
class Pooling : SN_Base::OperatorBase{

public:

	Pooling(const std::string& name, const std::string& node, std::map<std::string, std::string>& prms);

	~Pooling() = default;

	std::vector<std::string> Do(const SN_Base::learningParam&, const std::vector<OperatorBase*>& neighbOpr) override;
	
		
private:
		
	size_t kernel_ = 2;                                         ///< размер
		
	poolType poolType_ = poolType::max;                         ///< тип

	SN_Base::Tensor inFwTns_, inBwTns_;                         ///< тензор с сосед слоя 
		
	void load(std::map<std::string, std::string>& prms);
		
	void forward(SN_Base::Tensor* inTns);
	void backward(SN_Base::Tensor* inTns, const SN_Base::learningParam& lernPrm);

};