

#pragma once

#include "stdafx.h"

/// adaptive gradient method
void opt_adagrad(SN_Base::snFloat* dW, SN_Base::snFloat* ioWGr, SN_Base::snFloat* ioW, size_t sz, SN_Base::snFloat alpha = 0.001F, SN_Base::snFloat lambda = 0.F, SN_Base::snFloat eps = 1e-8F);

/// RMSprop
void opt_RMSprop(SN_Base::snFloat* dW, SN_Base::snFloat* ioWGr, SN_Base::snFloat* ioW, size_t sz, SN_Base::snFloat alpha = 0.001F, SN_Base::snFloat lambda = 0.F, SN_Base::snFloat mu = 0.9F, SN_Base::snFloat eps = 1e-8F);

/// adam
void opt_adam(SN_Base::snFloat* dW, SN_Base::snFloat* iodWPrev, SN_Base::snFloat* ioWGr, SN_Base::snFloat* ioW, size_t sz, SN_Base::snFloat alpha = 0.001F, SN_Base::snFloat lambda = 0.F, SN_Base::snFloat muWd = 0.9F, SN_Base::snFloat muGr = 0.9F, SN_Base::snFloat eps = 1e-8F);

/// SGD
void opt_sgd(SN_Base::snFloat* dW, SN_Base::snFloat* ioW, size_t sz, SN_Base::snFloat alpha = 0.001F, SN_Base::snFloat lambda = 0.F);

/// SGD with momentum
void opt_sgdMoment(SN_Base::snFloat* dW, SN_Base::snFloat* iodWPrev, SN_Base::snFloat* ioW, size_t sz, SN_Base::snFloat alpha = 0.01F, SN_Base::snFloat lambda = 0.F, SN_Base::snFloat mu = 0.9F);
