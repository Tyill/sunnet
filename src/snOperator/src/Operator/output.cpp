
#include "../stdafx.h"
#include "Output.h"

using namespace std;
using namespace SN_Base;



/// конец сети - оператор заглушка - ничего не должен делать!
Output::Output(const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(name, node, prms){
		
}

/// выполнить расчет
std::vector<std::string> Output::Do(const learningParam& lernPrm, const std::vector<OperatorBase*>& neighbOpr){
		
	if (neighbOpr.size() == 1){
		if (lernPrm.action == snAction::forward)
			baseOut_ = neighbOpr[0]->getOutput();
	}
	else{
		if (lernPrm.action == snAction::forward){

			inFwTns_ = *neighbOpr[0]->getOutput();

			int sz = neighbOpr.size();
			for (size_t i = 1; i < sz; ++i)
				inFwTns_ += *neighbOpr[i]->getOutput();

			baseOut_ = &inFwTns_;
		}
	}

	return std::vector<std::string>();
}


