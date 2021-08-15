//
// sunnet project
// Copyright (C) 2018 by Contributors <https://github.com/Tyill/sunnet>
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

namespace Test
{
    class Program
    {
        static bool loadImage(string imgPath, uint classCnt, List<List<string>> imgName, List<int> imgCntDir)
        {           
            for (int i = 0; i < classCnt; ++i){

                string dir = imgPath + i.ToString() + "/";
                if (!Directory.Exists(dir)) continue;

                imgName.Add(new List<string>());
                string[] files = Directory.GetFiles(dir);
                foreach (string s in files)
                {
                    imgName[i].Add(s);
                }
                imgCntDir.Add(files.Count());
            }
            bool ok = imgCntDir.Count == classCnt;
            foreach (int cnt in imgCntDir)
                if (cnt == 0) ok = false;
            
            return ok;
        }

        static void Main(string[] args)
        {           
            sn.Net snet = new sn.Net();

            string ver = snet.versionLib();
            Console.WriteLine("Version snlib " + ver);

            snet.addNode("Input", new sn.Input(), "C1")
                .addNode("C1", new sn.Convolution(15, 0), "C2")
                .addNode("C2", new sn.Convolution(15, 0), "P1")
                .addNode("P1", new sn.Pooling(), "FC1")
                .addNode("FC1", new sn.FullyConnected(128), "FC2")
                .addNode("FC2", new sn.FullyConnected(10), "LS")
                .addNode("LS", new sn.LossFunction(sn.lossType.type.softMaxToCrossEntropy), "Output");

            string imgPath = "c://cpp//sunnet//example//mnist//images//";

                      
            uint batchSz = 100, classCnt = 10, w = 28, h = 28; float lr = 0.001F;
            List<List<string>> imgName = new List<List<string>>();
            List<int> imgCntDir = new List<int>(10);
            Dictionary<string, Bitmap> images = new Dictionary<string, Bitmap>();

            if (!loadImage(imgPath, classCnt, imgName, imgCntDir))
            {
                Console.WriteLine("Error 'loadImage' path: " + imgPath);
                Console.ReadKey();
                return;
            }

            string wpath = "c://cpp//w.dat";
            if (snet.loadAllWeightFromFile(wpath))
                Console.WriteLine("Load weight ok path: " + wpath);
            else
                Console.WriteLine("Load weight err path: " + wpath);

          
            sn.Tensor inLayer = new sn.Tensor(new sn.snLSize(w, h, 1, batchSz));
            sn.Tensor targetLayer = new sn.Tensor(new sn.snLSize(classCnt, 1, 1, batchSz));
            sn.Tensor outLayer = new sn.Tensor(new sn.snLSize(classCnt, 1, 1, batchSz));
                  
            float accuratSumm = 0;
            for (int k = 0; k < 1000; ++k){

                targetLayer.reset();
                Random rnd = new Random();

                for (int i = 0; i < batchSz; ++i)
                {

                    // directory                   
                    int ndir = rnd.Next(0, (int)classCnt);
                    while (imgCntDir[ndir] == 0) ndir = rnd.Next(0, (int)classCnt);

                    // image
                    int nimg = rnd.Next(0, imgCntDir[ndir]);

                    // read
                    Bitmap img;
                    string fn = imgName[ndir][nimg];
                    if (images.ContainsKey(fn))
                        img = images[fn];
                    else
                    {
                        img = new Bitmap(fn);
                        images.Add(fn, img);
                    }

                    unsafe
                    {
                        float* refData = inLayer.data() + i * w * h;
                        int nr = img.Height, nc = img.Width;
                        System.Drawing.Imaging.BitmapData bmd = img.LockBits(new Rectangle(0, 0, img.Width, img.Height),
                            System.Drawing.Imaging.ImageLockMode.ReadWrite, img.PixelFormat);
                        IntPtr pt = bmd.Scan0;
                        for (int r = 0; r < nr; ++r)
                        {
                            for (int c = 0; c < nc; ++c)
                            {
                                refData[r * nc + c] = Marshal.ReadByte(pt);
                                pt += 4;
                            }
                        }
                        img.UnlockBits(bmd);

                        float* tarData = targetLayer.data() + classCnt * i;
                        tarData[ndir] = 1;
                    }
                }

                // training
                float accurat = 0;
                snet.training(lr, inLayer, outLayer, targetLayer, ref accurat);
                                
                // calc error
                int accCnt = 0;
                unsafe
                {
                    float* targetData = targetLayer.data();
                    float* outData = outLayer.data();
                    int bsz = (int)batchSz;
                    for (int i = 0; i < bsz; ++i)
                    {
                        float* refOutput = outData + i * classCnt;
                         
                        float maxval = refOutput[0];
                        int maxOutInx = 0;
                        for (int j = 1; j < classCnt; ++j){
                            if (refOutput[j] > maxval){
                                maxval = refOutput[j];
                                maxOutInx = j;
                            }
                        }

                        float* refTarget = targetData + i * classCnt;
                       
                        maxval = refTarget[0];
                        int maxTargInx = 0;
                        for (int j = 1; j < classCnt; ++j){
                            if (refTarget[j] > maxval){
                                maxval = refTarget[j];
                                maxTargInx = j;
                            }
                        }

                        if (maxTargInx == maxOutInx)
                            ++accCnt;
                    }
                }

                accuratSumm += (float)accCnt / batchSz;

                Console.WriteLine(k.ToString() + " accurate " + (accuratSumm / (k + 1)).ToString() + " " +
                    snet.getLastErrorStr());        
            }

            if (snet.saveAllWeightToFile(wpath))
                Console.WriteLine("Save weight ok path: " + wpath);
            else
                Console.WriteLine("Save weight err path: " + wpath);

            Console.ReadKey();
            return;           
        }
    }              
            
}
