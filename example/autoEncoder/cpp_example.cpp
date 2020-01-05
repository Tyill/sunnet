
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
            ++cnt;
        }

        imgCntDir[i] = cnt;
    }

    return true;
}

int main(int argc, char* argv[]){

    sn::Net snet;

    snet.addNode("Input", sn::Input(), "FC1")
        .addNode("FC1", sn::FullyConnected(256, sn::active::relu), "FC2")
        .addNode("FC2", sn::FullyConnected(128, sn::active::relu), "FC3")
        .addNode("FC3", sn::FullyConnected(32, sn::active::relu), "FC4")
        .addNode("FC4", sn::FullyConnected(128, sn::active::relu), "FC5")
        .addNode("FC5", sn::FullyConnected(256, sn::active::relu), "FC6")
        .addNode("FC6", sn::FullyConnected(784, sn::active::sigmoid), "LS")
        .addNode("LS", sn::LossFunction(sn::lossType::binaryCrossEntropy), "Output");

    string imgPath = "c://cpp//skyNet//example//autoEncoder//images//";

    int classCnt = 5, batchSz = 100, w = 28, h = 28;
    float lr = 0.001F;

    vector<vector<string>> imgName(classCnt);
    vector<int> imgCntDir(classCnt);
    map<string, cv::Mat> images;

    if (!loadImage(imgPath, classCnt, imgName, imgCntDir, images)){
        cout << "Error 'loadImage' path: " << imgPath << endl;
        system("pause");
        return -1;
    }

    //snet.loadAllWeightFromFile("c:\\cpp\\w.dat");


    sn::Tensor inLayer(sn::snLSize(w, h, 1, batchSz));
    sn::Tensor outLayer(sn::snLSize(w * h, 1, 1, batchSz));

    size_t sum_metric = 0;
    size_t num_inst = 0;
    float accuratSumm = 0;
    for (int k = 0; k < 1000; ++k){

        srand(clock());

        for (int i = 0; i < batchSz; ++i){

            // directory
            int ndir = rand() % classCnt;
            while (imgCntDir[ndir] == 0)
                ndir = rand() % classCnt;

            // image
            int nimg = rand() % imgCntDir[ndir];

            // read
            cv::Mat img;
            string nm = imgName[ndir][nimg];
            if (images.find(nm) != images.end())
                img = images[nm];
            else{
                img = cv::imread(imgPath + to_string(ndir) + "/" + nm, CV_LOAD_IMAGE_UNCHANGED);
                images[nm] = img;
            }

            float* refData = inLayer.data() + i * w * h;
   
            size_t nr = img.rows, nc = img.cols;
            for (size_t r = 0; r < nr; ++r){
                uchar* pt = img.ptr<uchar>(r);
                for (size_t c = 0; c < nc; ++c)
                    refData[r * nc + c] = pt[c] / 255.0;
            }
        }

        // training
        float accurat = 0;
        snet.training(lr,
                      inLayer,
                      outLayer,
                      inLayer,
                      accurat);

        float* refData = outLayer.data();

        cv::Mat img(w, h, CV_8U);
        for (size_t r = 0; r < h; ++r){
            uchar* pt = img.ptr<uchar>(r);
            for (size_t c = 0; c < w; ++c)
                pt[c] = refData[r * w + c] * 255.0;
        }

        cv::namedWindow("1", 0);
        cv::imshow("1", img);
        cv::waitKey(1);

        accuratSumm += accurat;

        cout << k << " accurate " << accuratSumm / k << " " << snet.getLastErrorStr() << endl;
    }
    
    snet.saveAllWeightToFile("c:\\cpp\\w.dat");

    system("pause");
    return 0;
}
