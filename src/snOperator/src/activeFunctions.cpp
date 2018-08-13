
#include "stdafx.h"

using namespace std;
using namespace SN_Base;

// fv - функция значения, df - производная функции


void fv_sigmoid(snFloat* ioVal, size_t sz){
	
	for (size_t i = 0; i < sz; ++i){
	
		ioVal[i] = 1.F / (1.F + std::exp(-ioVal[i]));
	}
}	
void df_sigmoid(snFloat* ioSigm, size_t sz){
	
	for (size_t i = 0; i < sz; ++i){

		ioSigm[i] = ioSigm[i] * (1.F - ioSigm[i]);
	}
}

void fv_relu(snFloat* ioVal, size_t sz){
	
	for (size_t i = 0; i < sz; ++i){

		ioVal[i] = ioVal[i] >= 0 ? ioVal[i] : 0;
	}
};
void df_relu(snFloat* ioRelu, size_t sz){
	
	for (size_t i = 0; i < sz; ++i){

		ioRelu[i] = ioRelu[i] >= 0 ? 1.F : 0.F;
	}
};

void fv_leakyRelu(snFloat* ioVal, size_t sz, snFloat minv){
	
	for (size_t i = 0; i < sz; ++i){

		ioVal[i] = ioVal[i] >= 0 ? ioVal[i] : minv * ioVal[i];
	}
}
void df_leakyRelu(snFloat* ioRelu, size_t sz, snFloat minv){
	
	for (size_t i = 0; i < sz; ++i){

		ioRelu[i] = ioRelu[i] >= 0 ? 1 : minv;
	}
}

void fv_elu(snFloat* ioVal, size_t sz, snFloat minv){
	
    for (size_t i = 0; i < sz; ++i){

		ioVal[i] = ioVal[i] >= 0 ? ioVal[i] : minv * (exp(ioVal[i]) - 1.F);
	}
}
void df_elu(snFloat* ioElu, size_t sz, snFloat minv){
	
	for (size_t i = 0; i < sz; ++i){

		ioElu[i] = ioElu[i] >= 0 ? 1 : ioElu[i] + minv;
	}
}
