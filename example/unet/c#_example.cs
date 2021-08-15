
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
        static bool loadImage(string imgPath, List<string> imgName)
        {

            string dir = imgPath;

            string[] files = Directory.GetFiles(dir);
            foreach (string s in files)
            {
                imgName.Add(s);
            }
            return imgName.Count > 0;
        }

        static void Main(string[] args)
        {
            sn.Net snet = new sn.Net();

            string ver = snet.versionLib();
            Console.WriteLine("Version snlib " + ver);

            snet.addNode("In", new sn.Input(), "C1")
                .addNode("C1", new sn.Convolution(10, -1), "C2")
                .addNode("C2", new sn.Convolution(10, 0), "P1 Crop1")
                .addNode("Crop1", new sn.Crop(new sn.rect(0, 0, 487, 487)), "Rsz1")
                .addNode("Rsz1", new sn.Resize(new sn.diap(0, 10), new sn.diap(0, 10)), "Conc1")
                .addNode("P1", new sn.Pooling(), "C3")

                .addNode("C3", new sn.Convolution(10, -1), "C4")
                .addNode("C4", new sn.Convolution(10, 0), "P2 Crop2")
                .addNode("Crop2", new sn.Crop(new sn.rect(0, 0, 247, 247)), "Rsz2")
                .addNode("Rsz2", new sn.Resize(new sn.diap(0, 10), new sn.diap(0, 10)), "Conc2")
                .addNode("P2", new sn.Pooling(), "C5")

                .addNode("C5", new sn.Convolution(10, 0), "C6")
                .addNode("C6", new sn.Convolution(10, 0), "DC1")
                .addNode("DC1", new sn.Deconvolution(10, 0), "Rsz3")
                .addNode("Rsz3", new sn.Resize(new sn.diap(0, 10), new sn.diap(10, 20)), "Conc2")

                .addNode("Conc2", new sn.Concat("Rsz2 Rsz3"), "C7")

                .addNode("C7", new sn.Convolution(10, 0), "C8")
                .addNode("C8", new sn.Convolution(10, 0), "DC2")
                .addNode("DC2", new sn.Deconvolution(10, 0), "Rsz4")
                .addNode("Rsz4", new sn.Resize(new sn.diap(0, 10), new sn.diap(10, 20)), "Conc1")

                .addNode("Conc1", new sn.Concat("Rsz1 Rsz4"), "C9")

                .addNode("C9", new sn.Convolution(10, 0), "C10");

            sn.Convolution convOut = new sn.Convolution(1, 0);
            convOut.act = new sn.active(sn.active.type.sigmoid);
            snet.addNode("C10", convOut, "LS")
                .addNode("LS", new sn.LossFunction(sn.lossType.type.binaryCrossEntropy), "Output");
                       
            string imgPath = "c://cpp//other//sunnet//example//unet//images//";
            string targPath = "c://cpp//other//sunnet//example//unet//labels//";


            uint batchSz = 3, w = 512, h = 512, wo = 483, ho = 483; float lr = 0.001F;
            List<string> imgName = new List<string>();
            List<string> targName = new List<string>();

            if (!loadImage(imgPath, imgName) ||
                !loadImage(targPath, targName))
            {
                Console.WriteLine("Error 'loadImage' path: " + imgPath);
                Console.ReadKey();
                return;
            }

            string wpath = "c:/cpp/w.dat";
            //  if (snet.loadAllWeightFromFile(wpath))
            //     Console.WriteLine("Load weight ok path: " + wpath);
            // else
            //     Console.WriteLine("Load weight err path: " + wpath);


            sn.Tensor inLayer = new sn.Tensor(new sn.snLSize(w, h, 1, batchSz));
            sn.Tensor targetLayer = new sn.Tensor(new sn.snLSize(wo, ho, 1, batchSz));
            sn.Tensor outLayer = new sn.Tensor(new sn.snLSize(wo, ho, 1, batchSz));

            float accuratSumm = 0;
            for (int k = 0; k < 1000; ++k)
            {

                targetLayer.reset();
                Random rnd = new Random();

                for (int i = 0; i < batchSz; ++i)
                {
                    // image
                    int nimg = rnd.Next(0, imgName.Count);

                    // read
                    Bitmap img = new Bitmap(imgName[nimg]);
                    unsafe
                    {
                        float* refData = inLayer.data() + i * w * h;
                        int nr = img.Height, nc = img.Width;
                        System.Drawing.Imaging.BitmapData bmd = img.LockBits(new Rectangle(0, 0, img.Width, img.Height),
                            System.Drawing.Imaging.ImageLockMode.ReadWrite, img.PixelFormat);

                        IntPtr ptData = bmd.Scan0;
                        for (int r = 0; r < nr; ++r)
                        {
                            for (int c = 0; c < nc; ++c)
                            {
                                refData[r * nc + c] = Marshal.ReadByte(ptData);

                                ptData += 4;
                            }
                        }
                        img.UnlockBits(bmd);


                        Bitmap imgTrg = new Bitmap(new Bitmap(targName[nimg]), new Size((int)wo, (int)ho));
                        nr = imgTrg.Height; nc = imgTrg.Width;

                        float* targData = targetLayer.data() + i * wo * ho;

                        System.Drawing.Imaging.BitmapData bmdTrg = imgTrg.LockBits(new Rectangle(0, 0, nc, nr),
                        System.Drawing.Imaging.ImageLockMode.ReadWrite, imgTrg.PixelFormat);

                        IntPtr ptTrg = bmdTrg.Scan0;
                        for (int r = 0; r < nr; ++r)
                        {
                            for (int c = 0; c < nc; ++c)
                            {
                                targData[r * nc + c] = (float)(Marshal.ReadByte(ptTrg) / 255.0);

                                ptTrg += 4;
                            }
                        }
                        imgTrg.UnlockBits(bmdTrg);

                    }
                }

                // training
                float accurat = 0;
                snet.training(lr, inLayer, outLayer, targetLayer, ref accurat);

                // calc error              
                accuratSumm += accurat;

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