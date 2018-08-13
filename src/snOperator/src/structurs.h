
#pragma once

#include "snBase/snBase.h"

/// тип ф-ии активации
enum class activeType{
	none = -1,
	sigmoid = 0,
	relu = 1,
	leakyRelu = 2,
	elu = 3,
};

/// тип инициализации весов
enum class weightInitType{
	uniform = 0,
	he = 1,
	lecun = 2,
	xavier = 3,
};

/// тип оптимизации весов
enum class optimizerType{
	sgd = 0,
	sgdMoment = 1,
	adagrad = 2,
	RMSprop = 3,
	adam = 4,
};

/// batchNorm
enum class batchNormType{
	none = -1,
	beforeActive = 0,
	postActive = 1,
};