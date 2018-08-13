
#include "snBase/snBase.h"
#include "random.h"

using namespace std;
using namespace SN_Base;

// инициализация весов

void wi_uniform(snFloat* ioW, size_t sz){
	
	rnd_uniform(ioW, sz, -1.F, 1.F);
};

void wi_xavier(snFloat* ioW, size_t sz, size_t fan_in, size_t fan_out){
	float_t wbase = std::sqrt(6.F / (fan_in + fan_out));

	rnd_uniform(ioW, sz, -wbase, wbase);
};

void wi_lecun(snFloat* ioW, size_t sz, size_t fan_out){
		
	float_t wbase = 1.F / std::sqrt(snFloat(fan_out));

	rnd_uniform(ioW, sz, -wbase, wbase);
}

void wi_he(snFloat* ioW, size_t sz, size_t fan_in){

	float_t sigma = std::sqrt(2.F / fan_in);

	rnd_gaussian(ioW, sz, 0.0F, sigma);
}

