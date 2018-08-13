
#pragma once

#include "snBase/snBase.h"
#include "skynet/skynet.h"

namespace SN_Opr{	
	
	/// создать оператор
    SN_Base::OperatorBase* createOperator(const std::string& fname, const std::string& node,
		std::map<std::string, std::string>& prms);	
	
	/// освободить оператор
	void freeOperator(SN_Base::OperatorBase*, const std::string& fname);
	
	/// задать статус callback общий
	bool setStatusCBack(SN_API::snStatusCBack, SN_API::snUData = nullptr);
}
