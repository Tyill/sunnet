
#pragma once

// инициализация весов

#include "snBase/snBase.h"
#include "random.h"

void wi_uniform(SN_Base::snFloat* ioW, size_t sz);

// 2010 Xavier Glorot
void wi_xavier(SN_Base::snFloat* ioW, size_t sz, size_t fan_in, size_t fan_out);

// 1998 Yann LeCun
void wi_lecun(SN_Base::snFloat* ioW, size_t sz, size_t fan_out);

// 2015 Kaiming He
void wi_he(SN_Base::snFloat* ioW, size_t sz, size_t fan_in);
