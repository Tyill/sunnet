// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include "snBase/snBase.h"
#include "snOperator/snOperator.h"


#ifdef SN_CPU
#include "Lib/OpenBLAS/include/cblas.h"
#else if SN_CUDA

#endif

void statusMess(const std::string&);