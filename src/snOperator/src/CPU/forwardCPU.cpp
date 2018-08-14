
#pragma once

#include "../stdafx.h"
#include "SNOperator/src/mathFunctions.h"
#include <omp.h>  

using namespace std;
using namespace SN_Base;


#ifdef SN_CPU

void fwdFullyConnected(size_t kernel, snSize insz, snFloat* input, snFloat* weight, snFloat* output){
		
	// Out = α * In * W + βC
	// In - матрица вход данных - значения с предыд слоя
	// W - матрица весов
	// Out - матрица выход данных
	cblas_sgemm(CBLAS_ORDER::CblasRowMajor,
		CBLAS_TRANSPOSE::CblasNoTrans,
		CBLAS_TRANSPOSE::CblasNoTrans,
		insz.n,                        // In, строк, кол-во изобр в батче
		kernel,                        // W, столбцов, кол-во скрытых нейронов 
		insz.w * insz.h * insz.d + 1,  // In, столбцов, В М - строк, кол-во вх нейронов - размер одного изображения из батча. (+1 - X0)                   
		1.0F,                          // α, коэф
		input,                         // In, вх данные - нейроны пришедшие с предыд слоя
		insz.w * insz.h * insz.d + 1,  // In, шаг до след X (X21 - X11) 
		weight,                        // W, веса
		kernel,                        // W, шаг до след W (W21 - W11) 
		0.0,                           // β, коэф
		output,                        // Out, выходные данные - нейроны для след слоя
		kernel);                       // Out, шаг до след Y (Y21 - Y11) 
}

void fwdConvolution(size_t kernel, size_t krnWidth, size_t krnHeight, size_t stride,
	snFloat* weight, snSize insz, snFloat* input, snSize outsz, snFloat* output){

	size_t wStepByD = krnWidth * krnHeight,              // шаг весов по входу
		   wStepByK = krnWidth * krnHeight * insz.d,     // шаг весов по выходу
		   inStepByD = insz.w * insz.h,                  // шаг вх слоя по входу
		   inStepByN = insz.w * insz.h * insz.d,         // шаг вх слоя по батчу
		   outStepByD = outsz.w * outsz.h,               // шаг вых слоя по выходу
		   outStepByN = outsz.w * outsz.h * outsz.d;     // шаг вых слоя по батчу

	size_t shareStepByN = insz.d + kernel;               // для локализации памяти
	snFloat* share = new snFloat[shareStepByN * insz.n];
	
	memset(output, 0, outStepByN * insz.n * sizeof(snFloat));
	
	// по батчу
#pragma omp parallel for
	for (int n = 0; n < insz.n; ++n){

		snFloat* inBuff = share + shareStepByN * n;
		snFloat* outBuff = share + insz.d + shareStepByN * n;

		for (size_t p = 0; p < outStepByD; ++p){
		
			size_t ox = p % outsz.w, oy = p / outsz.w,
				posW = ox * stride, posH = oy * stride;

			memset(outBuff, 0, kernel * sizeof(snFloat));

			// ядро свертки
			for (size_t c = 0; c < (krnWidth * krnHeight); ++c){

				size_t cx = c % krnWidth, cy = c / krnWidth;
				snFloat* pIn = input + (cx + posW) + (cy + posH) * insz.w + n * inStepByN;
				snFloat* pW = weight + cx + cy * krnWidth;

				for (size_t d = 0; d < insz.d; ++d){
					inBuff[d] = *pIn;
					pIn += inStepByD;
				}
			
				// по всем вых слоям
				for (size_t k = 0; k < kernel; ++k){
										
					// по всем вх слоям
					snFloat cout = 0;
					for (size_t d = 0; d < insz.d; ++d){
						cout += inBuff[d] * (*pW);
						pW += wStepByD;
					}
					pW += 1;           // bias;
					outBuff[k] += cout;
				}
			}

			snFloat* pOut = output + ox + oy * outsz.w + n * outStepByN;
			snFloat* pW = weight;

			// по всем вых слоям
			for (size_t k = 0; k < kernel; ++k){
				
				pW += wStepByK;

				*pOut += outBuff[k] + *(pW + k); // + bias
				
				pOut += outStepByD;
			}
		}		
	}
	
	delete[] share;
}

void fwdBatchNorm(snSize insz, snFloat* in, snFloat* out, batchNormParam prm){
 	
	size_t inSz = insz.w * insz.h * insz.d, bsz = insz.n;

	/// μ = 1/n * ∑x
	cblas_sgemv(CBLAS_ORDER::CblasRowMajor,
		CBLAS_TRANSPOSE::CblasTrans,
		bsz,                          // x, строк - размер батча
		inSz,                         // x, столбцов 
		1.F / bsz,                    // коэф
		in,                           // x, данные
		inSz,                         // x, шаг до след 
		prm.onc,                      // 1й вектор
		1,                            // 1й вектор, шаг движения по вектору
		0.0,                          // коэф
		prm.mean,                     // μ, результ
		1);                           // μ, шаг до след
				
	/// varce = sqrt(∑xx - mean^2 + e)
	for (size_t i = 0; i < inSz; ++i){

		snFloat* cin = in + i, srq = 0.F;
		for (size_t j = 0; j < bsz; ++j){
			srq += cin[0] * cin[0];
			cin += inSz;
		}
		prm.varce[i] = sqrt(srq / bsz - prm.mean[i] * prm.mean[i] + 0.0001F);
	}

	/// norm = (in - mean) / varce
	/// y = ^x * γ + β
	for (size_t i = 0; i < inSz; ++i){

		snFloat* cin = in + i, * cout = out + i, * norm = prm.norm + i,
			mean = prm.mean[i], varce = prm.varce[i], scale = prm.scale[i], schift = prm.schift[i];
		for (size_t j = 0; j < bsz; ++j){
						
			*norm = (*cin - mean) / varce;

			*cout = *norm * scale + schift;

			cin += inSz;
			cout += inSz;
			norm += inSz;			
		}
	}
}


#endif //#ifdef SN_CPU