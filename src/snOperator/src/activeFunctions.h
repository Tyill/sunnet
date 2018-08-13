
#pragma once

#include "stdafx.h"

// fv - функция значения, df - производная функции

void fv_sigmoid(SN_Base::snFloat* ioVal, size_t sz);
void df_sigmoid(SN_Base::snFloat* inSigm, size_t sz);

void fv_relu(SN_Base::snFloat* ioVal, size_t sz);
void df_relu(SN_Base::snFloat* inRelu, size_t sz);

void fv_leakyRelu(SN_Base::snFloat* ioVal, size_t sz, SN_Base::snFloat minv = 0.01F);
void df_leakyRelu(SN_Base::snFloat* inRelu, size_t sz, SN_Base::snFloat minv = 0.01F);

void fv_elu(SN_Base::snFloat* ioVal, size_t sz, SN_Base::snFloat minv = 0.01F);
void df_elu(SN_Base::snFloat* inElu, size_t sz, SN_Base::snFloat minv = 0.01F);
