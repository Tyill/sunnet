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

#define PROFILE_START double ctm = omp_get_wtime(); 
#define PROFILE_END(func) SN_PRINTMESS(std::string("Profile ") + func + " " + std::to_string(omp_get_wtime() - ctm)); ctm = omp_get_wtime(); 

#ifdef SN_CPU
#include "Lib/OpenBLAS/cblas.h"
#else if SN_CUDA

#endif

void statusMess(const std::string&);