
#include "stdafx.h"

using namespace std;
using namespace SN_Base;

/// adaptive gradient method
void opt_adagrad(snFloat* dW, snFloat* ioWGr, snFloat* ioW, size_t sz, snFloat alpha, snFloat lambda, snFloat eps){

	for (size_t i = 0; i < sz; ++i){
		ioWGr[i] += dW[i] * dW[i];
		ioW[i] -= alpha * (dW[i] + ioW[i] * lambda) / (std::sqrt(ioWGr[i]) + eps);
	}
}

/// RMSprop
void opt_RMSprop(snFloat* dW, snFloat* ioWGr, snFloat* ioW, size_t sz, snFloat alpha, snFloat lambda, snFloat mu, snFloat eps){
   
	for (size_t i = 0; i < sz; ++i){    
		ioWGr[i] = ioWGr[i] * mu  + (1.F - mu) * dW[i] * dW[i];
		ioW[i] -= alpha * (dW[i] + ioW[i] * lambda) / std::sqrt(ioWGr[i] + eps);
    }
}

/// adam
void opt_adam(snFloat* dW, snFloat* iodWPrev, snFloat* ioWGr, snFloat* ioW, size_t sz, snFloat alpha, snFloat lambda, snFloat mudW, snFloat muGr, snFloat eps){

	for (size_t i = 0; i < sz; ++i){
		
		iodWPrev[i] = iodWPrev[i] * mudW - (1.F - mudW) * alpha * (dW[i] + ioW[i] * lambda);
		
		ioWGr[i] = ioWGr[i] * muGr + (1.F - muGr) * dW[i] * dW[i];
			
		ioW[i] += iodWPrev[i] / std::sqrt(ioWGr[i] + eps);
	}
}

/// SGD without momentum
void opt_sgd(snFloat* dW, snFloat* ioW, size_t sz, snFloat alpha, snFloat lambda){
    for(size_t i = 0; i < sz; ++i){
		ioW[i] -= alpha * (dW[i] + lambda * ioW[i]);
    }
}

/// SGD with momentum
void opt_sgdMoment(snFloat* dW, snFloat* iodWPrev, snFloat* ioW, size_t sz, snFloat alpha, snFloat lambda, snFloat mu){
	
	for(size_t i = 0; i < sz; ++i){
		iodWPrev[i] = mu * iodWPrev[i] - alpha * (dW[i] + ioW[i] * lambda);
		ioW[i] += iodWPrev[i];
    }
}