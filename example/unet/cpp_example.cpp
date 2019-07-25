
#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <map>
#include <filesystem>

#include "../cpp/snNet.h"
#include "../cpp/snTensor.h"
#include "../cpp/snOperator.h"

#include "Lib/OpenCV_3.3.0/opencv2/core/core_c.h"
#include "Lib/OpenCV_3.3.0/opencv2/core/core.hpp"
#include "Lib/OpenCV_3.3.0/opencv2/imgproc/imgproc_c.h"
#include "Lib/OpenCV_3.3.0/opencv2/imgproc/imgproc.hpp"
#include "Lib/OpenCV_3.3.0/opencv2/highgui/highgui_c.h"
#include "Lib/OpenCV_3.3.0/opencv2/highgui/highgui.hpp"

using namespace std;
namespace sn = SN_API;

bool loadImage(string& imgPath, int classCnt, vector<string>& imgName, int& imgCntDir){
      
    namespace fs = std::tr2::sys;

    if (!fs::exists(fs::path(imgPath))) return false;

    fs::directory_iterator it(imgPath); int cnt = 0;
    while (it != fs::directory_iterator()){

        fs::path p = it->path();
        if (fs::is_regular_file(p) && (p.extension() == ".png")){

            imgName.push_back(p.filename());
        }
        ++it;
        ++cnt;
    }

    imgCntDir = cnt;
   
    return true;
}

int main(int argc, char* argv[]){
       
    sn::Net snet;   
 
    snet.addNode("In", sn::Input(), "C1")
        .addNode("C1", sn::Convolution(10, -1), "C2")
        .addNode("C2", sn::Convolution(10, 0), "P1 Crop1")
        .addNode("Crop1", sn::Crop(sn::rect(0, 0, 487, 487)), "Rsz1")
        .addNode("Rsz1", sn::Resize(sn::diap(0, 10), sn::diap(0, 10)), "Conc1")
        .addNode("P1", sn::Pooling(), "C3")

        .addNode("C3", sn::Convolution(10, -1), "C4")
        .addNode("C4", sn::Convolution(10, 0), "P2 Crop2")
        .addNode("Crop2", sn::Crop(sn::rect(0, 0, 247, 247)), "Rsz2")
        .addNode("Rsz2", sn::Resize(sn::diap(0, 10), sn::diap(0, 10)), "Conc2")
        .addNode("P2", sn::Pooling(), "C5")

        .addNode("C5", sn::Convolution(10, 0), "C6")
        .addNode("C6", sn::Convolution(10, 0), "DC1")
        .addNode("DC1", sn::Deconvolution(10), "Rsz3")
        .addNode("Rsz3", sn::Resize(sn::diap(0, 10), sn::diap(10, 20)), "Conc2")

        .addNode("Conc2", sn::Concat("Rsz2 Rsz3"), "C7")

        .addNode("C7", sn::Convolution(10, 0), "C8")
        .addNode("C8", sn::Convolution(10, 0), "DC2")
        .addNode("DC2", sn::Deconvolution(10), "Rsz4")
        .addNode("Rsz4", sn::Resize(sn::diap(0, 10), sn::diap(10, 20)), "Conc1")

        .addNode("Conc1", sn::Concat("Rsz1 Rsz4"), "C9")

        .addNode("C9", sn::Convolution(10, 0), "C10");

    sn::Convolution convOut(1, 0);
    convOut.act = sn::active::sigmoid;
    snet.addNode("C10", convOut, "LS")
        .addNode("LS", sn::LossFunction(sn::lossType::binaryCrossEntropy), "Output");
    
    string imgPath = "c://cpp//skyNet//example//unet//images//";
    string labelPath = "c://cpp//skyNet//example//unet//labels//";

    int batchSz = 10, w = 512, h = 512, wo = 483, ho = 483; float lr = 0.001F;
    vector<string> imgName;
    int imgCntDir = 0;
       
    if (!loadImage(imgPath, 1, imgName, imgCntDir)){
        cout << "Error loadImage path: " << imgPath << endl;
        system("pause");
        return -1;
    }

    sn::Tensor inLayer(sn::snLSize(w, h, 1, batchSz));
    sn::Tensor targetLayer(sn::snLSize(wo, ho, 1, batchSz));
    sn::Tensor outLayer(sn::snLSize(wo, ho, 1, batchSz));
       
    size_t sum_metric = 0;
    size_t num_inst = 0;
    float accuratSumm = 0;
    for (int k = 0; k < 1000; ++k){
               
        srand(clock());
                
        for (int i = 0; i < batchSz; ++i){
                       
            // image
            int nimg = rand() % imgCntDir;

            // read
            string nm = imgName[nimg];
            cv::Mat img = cv::imread(imgPath + nm, CV_LOAD_IMAGE_UNCHANGED);

            float* refData = inLayer.data() + i * w * h;
           
            for (size_t r = 0; r < h; ++r){
                uchar* pt = img.ptr<uchar>(r);
                for (size_t c = 0; c < w; c += 1){
                    refData[r * w + c] = pt[c];                   
                }
            } 

            cv::Mat imgLabel = cv::imread(labelPath + nm, CV_LOAD_IMAGE_UNCHANGED);

            cv::resize(imgLabel, imgLabel, cv::Size(wo, ho));

            refData = targetLayer.data() + i * wo * ho;

            for (size_t r = 0; r < ho; ++r){
                uchar* pt = imgLabel.ptr<uchar>(r);
                for (size_t c = 0; c < wo; c += 1){
                    refData[r * wo + c] = pt[c] / 255.;
                }
            }

        }

        // training
        float accurat = 0;
        snet.training(lr,
            inLayer,
            outLayer,
            targetLayer,
            accurat);

        // calc error                     
        accuratSumm += accurat;

        cout << k << " accurate " << accuratSumm / (k + 1) << " " << snet.getLastErrorStr() << endl;        
    }
       
    system("pause");
    return 0;
}