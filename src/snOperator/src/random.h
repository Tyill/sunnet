
#pragma once

#include <random>

#include "snBase/snBase.h"

void rnd_uniform(SN_Base::snFloat* ioVal, size_t sz, SN_Base::snFloat min, SN_Base::snFloat max);

void rnd_gaussian(SN_Base::snFloat* ioVal, size_t sz, SN_Base::snFloat mean, SN_Base::snFloat sigma);


