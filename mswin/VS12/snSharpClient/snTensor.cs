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
    public struct snLSize
    {
        public UInt64 w, h, ch, bsz;

        public snLSize(UInt64 w_ = 0, UInt64 h_ = 0, UInt64 ch_ = 0, UInt64 bsz_ = 0)
        {
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
       
        public Tensor(snLSize lsz){
        
            lsz_ = lsz;

            int sz = (int)(lsz.w * lsz.h * lsz.ch * lsz.bsz * sizeof(float));
            if (sz > 0)
                data_ = (float*)Marshal.AllocHGlobal(sz);
        }

        ~Tensor()
        {
            if (data_ != null)
                Marshal.FreeHGlobal((IntPtr)data_);
        }

        [DllImport("msvcrt.dll", EntryPoint = "memcpy", CallingConvention = CallingConvention.Cdecl, SetLastError = false)]
        public static extern IntPtr MemCopy(IntPtr dest, IntPtr src, UInt64 sz);

        public Tensor(snLSize lsz, float* data)
        {

            lsz_ = lsz;

            UInt64 sz = lsz.w * lsz.h * lsz.ch * lsz.bsz * sizeof(float);
            if (sz > 0)
            {
                data_ = (float*)Marshal.AllocHGlobal((int)sz);
                MemCopy((IntPtr)data_, (IntPtr)data, sz);
            }
        }
             
        [DllImport("msvcrt.dll", EntryPoint = "memset", CallingConvention = CallingConvention.Cdecl, SetLastError = false)]
        public static extern IntPtr MemSet(IntPtr dest, int c, UInt64 count);
        
        public void reset(){

            UInt64 sz = (UInt64)(lsz_.w * lsz_.h * lsz_.ch * lsz_.bsz * sizeof(float));
            MemSet((IntPtr)data_, 0, sz);
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
