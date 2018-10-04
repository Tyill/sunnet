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
#pragma once


#include <cstdint>
#include <string>
#include <vector>
#include "../src/skynet/skyNet.h"

namespace SN_API{
      
    class Tensor{
        
    public:

        Tensor(const snLSize& lsz = snLSize(), std::vector<snFloat>& data = std::vector<snFloat>()){
        
            lsz_ = lsz;
            data_ = data;
        };

        Tensor(const snLSize& lsz, snFloat* data){

            lsz_ = lsz; 
            size_t sz = lsz.w * lsz.h * lsz.ch * lsz.bsz;
         
            data_.resize(sz);

            memcpy(data_.data(), data, sz * sizeof(snFloat));
        };

        ~Tensor(){};
        
        bool addChannel(uint32_t w, uint32_t h, std::vector<snFloat>& data){

            if ((w != lsz_.w) || (h != lsz_.h)) return false;

            size_t csz = data_.size();
            data_.resize(csz + w * h);
            memcpy(data_.data() + csz, data.data(), w * h * sizeof(snFloat));

            ++chsz_;
            if (chsz_ == lsz_.ch){
                chsz_ = 0;
                ++lsz_.bsz;
            }

            return true;
        }

        bool addChannel(uint32_t w, uint32_t h, snFloat* data){

            if ((w != lsz_.w) || (h != lsz_.h)) return false;

            size_t csz = data_.size();
            data_.resize(csz + w * h);
            memcpy(data_.data() + csz, data, w * h * sizeof(snFloat));

            ++chsz_;
            if (chsz_ == lsz_.ch){
                chsz_ = 0;
                ++lsz_.bsz;
            }

            return true;
        }
                
        snFloat* data(){

            return data_.data();
        }

        snLSize size(){

            return lsz_;
        }

    private:
        size_t chsz_;
        snLSize lsz_;
        std::vector<snFloat> data_;
     
    };    
}
