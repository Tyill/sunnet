// FNOperator.cpp : Defines the exported functions for the DLL application.
//

#include "snOperator/snOperator.h"
#include "snBase/snBase.h"
#include "Operator/fullyConnected.h"
#include "Operator/convolution.h"
#include "Operator/input.h"
#include "Operator/output.h"
#include "Operator/lossFunction.h"

SN_API::snStatusCBack g_sts = nullptr;
SN_API::snUData g_ud = nullptr;

void statusMess(const std::string& mess){

	if (g_sts) g_sts(mess.c_str(), g_ud);
}

namespace SN_Opr{
	
	SN_Base::OperatorBase* createOperator(const std::string& fname, const std::string& node,
		std::map<std::string, std::string>& prms){

		SN_Base::OperatorBase* ret = nullptr;

		if (fname == "Input")               ret = (SN_Base::OperatorBase*)new Input(fname, node, prms);
		else if (fname == "Output")         ret = (SN_Base::OperatorBase*)new Output(fname, node, prms);
		else if (fname == "FullyConnected") ret = (SN_Base::OperatorBase*)new FullyConnected(fname, node, prms);
		else if (fname == "LossFunction")   ret = (SN_Base::OperatorBase*)new LossFunction(fname, node, prms);
		else if (fname == "Convolution")    ret = (SN_Base::OperatorBase*)new Convolution(fname, node, prms);
		
		return ret;
	}

	/// освободить оператор
	void freeOperator(SN_Base::OperatorBase* opr, const std::string& fname){

		if (opr){
			if (fname == "Input")               delete (Input*)opr;
			else if (fname == "Output")         delete (Output*)opr;
			else if (fname == "FullyConnected") delete (FullyConnected*)opr;
			else if (fname == "LossFunction")   delete (LossFunction*)opr;
			else if (fname == "Convolution")    delete (Convolution*)opr;
		}
	}
		
	/// задать статус callback
	bool setStatusCBack(SN_API::snStatusCBack sts, SN_API::snUData ud){

		g_sts = sts;
		g_ud = ud;

		return true;
	}
}
