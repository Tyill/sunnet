
#include "../stdafx.h"
#include "Input.h"

using namespace std;
using namespace SN_Base;

/// начало сети - оператор заглушка - ничего не должен делать!
Input::Input(const string& name, const string& node, std::map<std::string, std::string>& prms) :
OperatorBase(name, node, prms){

	
}

/// задать аргументы для расчета
bool Input::setInput(SN_Base::Tensor* args){
	baseOut_ = args;
	return true;
}

/// выполнить расчет
std::vector<std::string> Input::Do(const learningParam&, const std::vector<OperatorBase*>& neighbOpr){
		
	
	return std::vector<std::string>();
}