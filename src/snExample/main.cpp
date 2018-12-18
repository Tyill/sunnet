
#ifndef CV_VERSION

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

bool loadImage(string& imgPath, int classCnt, vector<vector<string>>& imgName, vector<int>& imgCntDir, map<string, cv::Mat>& images){
    
    for (int i = 0; i < classCnt; ++i){

        namespace fs = std::tr2::sys;

        if (!fs::exists(fs::path(imgPath + to_string(i) + "/"))) continue;

        fs::directory_iterator it(imgPath + to_string(i) + "/"); int cnt = 0;
        while (it != fs::directory_iterator()){

            fs::path p = it->path();
            if (fs::is_regular_file(p) && (p.extension() == ".png")){

                imgName[i].push_back(p.filename());
            }
            ++it;
            ++cnt; if (cnt > 1000) break;
        }

        imgCntDir[i] = cnt;
    }

    return true;
}

int main(int argc, char* argv[]){
       
    sn::Net snet;
  
    snet.addNode("Input", sn::Input(), "C1")
        .addNode("C1", sn::Convolution(15, 0, sn::calcMode::CPU), "C2")
        .addNode("C2", sn::Convolution(15, 0, sn::calcMode::CPU), "P1")
        .addNode("P1", sn::Pooling(sn::calcMode::CPU), "FC1")
        .addNode("FC1", sn::FullyConnected(128, sn::calcMode::CPU), "FC2")
        .addNode("FC2", sn::FullyConnected(10, sn::calcMode::CPU), "LS")
        .addNode("LS", sn::LossFunction(sn::lossType::softMaxToCrossEntropy), "Output");

    string imgPath = "c://C++//skyNet//example//mnist//images//";
    
    int batchSz = 100, classCnt = 10, w = 28, h = 28; float lr = 0.001F;
    vector<vector<string>> imgName(classCnt);
    vector<int> imgCntDir(classCnt);
    map<string, cv::Mat> images;
       
    if (!loadImage(imgPath, classCnt, imgName, imgCntDir, images)){
        cout << "Error 'loadImage' path: " << imgPath << endl;
        system("pause");
        return -1;
    }

   // snet.loadAllWeightFromFile("c:/C++/w.dat");
      
    sn::Tensor inLayer(sn::snLSize(w, h, 1, batchSz));
    sn::Tensor targetLayer(sn::snLSize(classCnt, 1, 1, batchSz));
    sn::Tensor outLayer(sn::snLSize(classCnt, 1, 1, batchSz));
       
    size_t sum_metric = 0;
    size_t num_inst = 0;
    float accuratSumm = 0;
    for (int k = 0; k < 10000; ++k){

        targetLayer.clear();
       
        srand(clock());
                
        for (int i = 0; i < batchSz; ++i){

            // directory
            int ndir = rand() % classCnt;
            while (imgCntDir[ndir] == 0) ndir = rand() % classCnt;

            // image
            int nimg = rand() % imgCntDir[ndir];

            // read
            cv::Mat img; string nm = imgName[ndir][nimg];
            if (images.find(nm) != images.end())
                img = images[nm];
            else{
                img = cv::imread(imgPath + to_string(ndir) + "/" + nm, CV_LOAD_IMAGE_UNCHANGED);
                images[nm] = img;
            }

            float* refData = inLayer.data() + i * w * h;
           
            double mean = cv::mean(img)[0];
            size_t nr = img.rows, nc = img.cols;
            for (size_t r = 0; r < nr; ++r){
                uchar* pt = img.ptr<uchar>(r);
                for (size_t c = 0; c < nc; ++c)
                    refData[r * nc + c] = pt[c];
            } 

            float* tarData = targetLayer.data() + classCnt * i;

            tarData[ndir] = 1;
        }

        // training
        float accurat = 0;
        snet.training(lr,
            inLayer,
            outLayer,
            targetLayer,
            accurat);

        // calc error
        sn::snFloat* targetData = targetLayer.data();
        sn::snFloat* outData = outLayer.data();
        size_t accCnt = 0, bsz = batchSz;
        for (int i = 0; i < bsz; ++i){

            float* refTarget = targetData + i * classCnt;
            float* refOutput = outData + i * classCnt;

            int maxOutInx = distance(refOutput, max_element(refOutput, refOutput + classCnt)),
                maxTargInx = distance(refTarget, max_element(refTarget, refTarget + classCnt));

            if (maxTargInx == maxOutInx)
                ++accCnt;
        }
              
        accuratSumm += (accCnt * 1.F) / bsz;

        cout << k << " accurate " << accuratSumm / k << " " << snet.getLastErrorStr() << endl;        
    }
       
    system("pause");
    return 0;
}

#else


#include "../cpp/snNet.h"
#include "../cpp/snTensor.h"
#include "../cpp/snOperator.h"
#include <iostream>

using namespace std;
namespace sn = SN_API;

int main(int argc, char* argv[]){

    sn::Net snet;

    snet.addNode("Input", sn::Input(), "C1")
        .addNode("C1", sn::Convolution(15, 0, sn::calcMode::CUDA), "C2")
        .addNode("C2", sn::Convolution(15, 0, sn::calcMode::CUDA), "P1")
        .addNode("P1", sn::Pooling(sn::calcMode::CUDA), "FC1")
        .addNode("FC1", sn::FullyConnected(128, sn::calcMode::CUDA), "FC2")
        .addNode("FC2", sn::FullyConnected(10, sn::calcMode::CUDA), "LS")
        .addNode("LS", sn::LossFunction(sn::lossType::softMaxToCrossEntropy), "Output");

    cout << "Hello " <<  SN_API::versionLib() << endl;

}

#endif
