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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace SN_API
{
    /// <summary>
    /// layer size
    /// </summary>
    public unsafe struct snLSize
    {
        public Int64 w = 0, h = 0, ch = 0, bsz = 0; 
 
        public snLSize(uint w_ = 0, uint h_ = 0, uint ch_ = 0, uint bsz_ = 0){
             w = w_;
             h = h_;
             ch = ch_;
             bsz = bsz_;
        }
    }

    /// <summary>
    /// Tensor
    /// </summary>
    public unsafe class Tensor{
       
        public Tensor(snLSize lsz, float* data = null){
        
            lsz_ = lsz;

            data_ = data;
        }
                           
        public void clear(){

         //   data_.;
        }

        public float* data(){

            return data_;
        }

        public snLSize size(){

            return lsz_;
        }

       private snLSize lsz_;
       private float* data_ = null;
     
    };    
}
