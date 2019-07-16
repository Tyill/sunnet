
#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "../stdafx.h"

using namespace SN_Base;


void dropOut(bool isLern, SN_Base::snFloat dropOut, const SN_Base::snSize& outsz, SN_Base::snFloat* out){

   if (isLern){
        size_t sz = size_t(outsz.size() * dropOut);
        vector<int> rnd(sz);
        rnd_uniformInt(rnd.data(), sz, 0, int(outsz.size()));

        for (auto i : rnd) out[i] = 0;
    }
    else{
        size_t sz = outsz.size();
        for (size_t i = 0; i < sz; ++i)
            out[i] *= (1.F - dropOut);
    }

}