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
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Drawing;
using System.IO;
using sn = SN_API;
using SN_API;

namespace Test
{
    class Program
    {
      
        static void Main(string[] args)
        {          
                      
            // using python for create file 'resNet50Weights.dat' as: 
            // CMD: cd c:\cpp\other\skyNet\example\resnet50\
            // CMD: python createNet.py  

            string arch = File.ReadAllText(@"c:\cpp\other\skyNet\example\resnet50\resNet50Struct.json", Encoding.UTF8);

            sn.Net snet = new sn.Net(arch, @"c:\cpp\other\skyNet\example\resnet50\resNet50Weights.dat");

            if (snet.getLastErrorStr().Count() > 0)
            {
                Console.WriteLine("Error loadAllWeightFromFile: " + snet.getLastErrorStr());
                Console.ReadKey();
                return;
            }

            string imgPath = @"c:\cpp\other\skyNet\example\resnet50\images\elephant.jpg";
                                  
            int classCnt = 1000, w = 224, h = 224;

            sn.Tensor inLayer = new sn.Tensor(new snLSize((UInt64)w, (UInt64)h, 3, 1));
            sn.Tensor outLayer = new sn.Tensor(new snLSize((UInt64)classCnt, 1, 1, 1));
              
            // read
            Bitmap img = new Bitmap(Image.FromFile(imgPath), new Size(w, h));
                        
            unsafe
            {
                float* refData = inLayer.data();
                                
                System.Drawing.Imaging.BitmapData bmd = img.LockBits(new Rectangle(0, 0, img.Width, img.Height),
                    System.Drawing.Imaging.ImageLockMode.ReadWrite, img.PixelFormat);
                
                // B
                IntPtr pt = bmd.Scan0;
                for (int r = 0; r < h; ++r)
                {
                    for (int c = 0; c < w; ++c)
                    {
                        refData[r * w + c] = Marshal.ReadByte(pt + 3);
                        pt += 4;
                    }
                }
              
                // G
                pt = bmd.Scan0;
                refData += h * w;
                for (int r = 0; r < h; ++r)
                {
                    for (int c = 0; c < w; ++c)
                    {
                        refData[r * w + c] = Marshal.ReadByte(pt + 2);
                        pt += 4;
                    }
                }

                // R
                pt = bmd.Scan0;
                refData += h * w;
                for (int r = 0; r < h; ++r)
                {
                    for (int c = 0; c < w; ++c)
                    {
                        refData[r * w + c] = Marshal.ReadByte(pt + 1);
                        pt += 4;
                    }
                }
                              
                img.UnlockBits(bmd);                
            }
              
            // training
            snet.forward(false, inLayer, outLayer);

            float maxval = 0;
            int maxOutInx = 0;

            unsafe{

                float* refOutput = outLayer.data();
                         
                maxval = refOutput[0];
                for (int j = 1; j < classCnt; ++j){
                    if (refOutput[j] > maxval){
                        maxval = refOutput[j];
                        maxOutInx = j;
                    }
                }
             }

            // for check: c:\cpp\other\skyNet\example\resnet50\imagenet_class_index.json
                
            Console.WriteLine("inx " + maxOutInx.ToString() + " accurate " + maxval.ToString() + " " + snet.getLastErrorStr());
            Console.ReadKey();
            return;
        }
    }       
}
