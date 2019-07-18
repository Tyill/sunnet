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

#include "snBase/snBase.h"

namespace SN_SIMD{
     

    /// @param[in] Mask := [1, 3..9], Stride := [1, ..), Dilate := [1, ..) 
    /// @param[in] buffMem size := M * M * insz.d * outsz.w * outsz.h
    /// @return true - ok
    bool convolutionFWD(size_t M, size_t S, size_t D,        
        const SN_Base::snFloat* weight,
        const SN_Base::snSize& insz, const SN_Base::snFloat* input,
        const SN_Base::snSize& outsz, SN_Base::snFloat* output,
        SN_Base::snFloat* buffMem);

    /// @param[in] Mask := [1, 3..9], Stride := [1, ..), Dilate := [1, ..) 
    /// @return true - ok
    bool convolutionBWD_GW(size_t M, size_t S, size_t D,
        const SN_Base::snFloat* weight,
        const SN_Base::snSize& insz, const SN_Base::snFloat* input,
        const SN_Base::snSize& outsz, const SN_Base::snFloat* gradIn, SN_Base::snFloat* gradOut, SN_Base::snFloat* dWeightOut);

    /// @param[in] Mask := [1, 3..9], Stride := [1, ..), Dilate := [1, ..) 
    /// @return true - ok
    bool convolutionBWD_G(size_t M, size_t S, size_t D,
        const SN_Base::snFloat* weight, const SN_Base::snSize& insz, const SN_Base::snSize& outsz,
        const SN_Base::snFloat* gradIn, SN_Base::snFloat* gradOut);
};
