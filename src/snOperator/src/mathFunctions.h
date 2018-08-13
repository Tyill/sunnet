
#pragma once

#include "snBase/snBase.h"

struct batchNormParam{
	SN_Base::snFloat* norm;      ///< нормирован вх значения
	SN_Base::snFloat* mean;      ///< среднее вх значений
	SN_Base::snFloat* varce;     ///< дисперсия вх значений
	SN_Base::snFloat* scale;     ///< коэф γ
	SN_Base::snFloat* dScale;    ///< dγ
	SN_Base::snFloat* schift;    ///< коэф β
	SN_Base::snFloat* dSchift;   ///< dβ
	SN_Base::snFloat* onc;       ///< 1й вектор
	SN_Base::snFloat lr = 0.001F; ///< коэф для изменения γ и β
};

void fwdBatchNorm(SN_Base::snSize insz,
	              SN_Base::snFloat* in,
				  SN_Base::snFloat* out,
	              batchNormParam);

void bwdBatchNorm(SN_Base::snSize insz, 
	              SN_Base::snFloat* gradIn,
				  SN_Base::snFloat* gradOut,
				  batchNormParam);
