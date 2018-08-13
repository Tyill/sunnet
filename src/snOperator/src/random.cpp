
#include <random>
#include <ctime>
#include "snBase/snBase.h"

// генераторы случ значений

using namespace std;
using namespace SN_Base;

void rnd_uniform(SN_Base::snFloat* ioVal, size_t sz, snFloat min, snFloat max) {
	std::uniform_real_distribution<snFloat> dst(min, max);
	
	std::mt19937 rnd_generator(clock());
	for (size_t i = 0; i < sz; ++i)
		ioVal[i] = dst(rnd_generator);
}

void rnd_gaussian(SN_Base::snFloat* ioVal, size_t sz, SN_Base::snFloat mean, SN_Base::snFloat sigma) {
	
	std::mt19937 rnd_generator(clock());
	std::normal_distribution<snFloat> dst(mean, sigma);
	for (size_t i = 0; i < sz; ++i)
		ioVal[i] = dst(rnd_generator);
}