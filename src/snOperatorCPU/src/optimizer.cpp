//
// SkyNet Project
// Copyright (C) 2018 by Contributors <https://github.com/Tyill/skynet>
//
// This code is licensed under the MIT License.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
#include "stdafx.h"
#include "optimizer.h"
#include "structurs.h"


using namespace std;
using namespace SN_Base;

void optimizer(snFloat* dWeight, snFloat* dWPrev, snFloat* dWGrad, snFloat* weight, size_t wsz, snFloat alpha, snFloat lambda, snFloat mudW, snFloat muGr, optimizerType otype){

    switch (otype){
    case optimizerType::sgd:       opt_sgd(dWeight, weight, wsz, alpha, lambda); break;
    case optimizerType::sgdMoment: opt_sgdMoment(dWeight, dWPrev, weight, wsz, alpha, lambda, mudW); break;
    case optimizerType::RMSprop:   opt_RMSprop(dWeight, dWGrad, weight, wsz, alpha, lambda, muGr); break;
    case optimizerType::adagrad:   opt_adagrad(dWeight, dWGrad, weight, wsz, alpha, lambda); break;
    case optimizerType::adam:      opt_adam(dWeight, dWPrev, dWGrad, weight, wsz, alpha, lambda, mudW, muGr); break;
    default: break;
    }     
}

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