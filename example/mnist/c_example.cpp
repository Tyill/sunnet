#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <map>
#include <filesystem>

#include "../sunnet/sunnet.h"

#include "Lib/OpenCV_3.3.0/opencv2/core/core_c.h"
#include "Lib/OpenCV_3.3.0/opencv2/core/core.hpp"
#include "Lib/OpenCV_3.3.0/opencv2/imgproc/imgproc_c.h"
#include "Lib/OpenCV_3.3.0/opencv2/imgproc/imgproc.hpp"
#include "Lib/OpenCV_3.3.0/opencv2/highgui/highgui_c.h"
#include "Lib/OpenCV_3.3.0/opencv2/highgui/highgui.hpp"

using namespace std;

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

int main(int argc, char* argv[])
{
    namespace sn = SN_API;

    stringstream ss;

    ss << "{"

        "\"BeginNet\":"
        "{"
        "\"NextNodes\":\"C1\""
        "},"

        "\"Nodes\":"
        "["

        "{"
        "\"NodeName\":\"C1\","
        "\"NextNodes\":\"C2\","
        "\"OperatorName\":\"Convolution\","
        "\"OperatorParams\":{\"kernel\":\"15\"}"
        "},"

        "{"
        "\"NodeName\":\"C2\","
        "\"NextNodes\":\"P1\","
        "\"OperatorName\":\"Convolution\","
        "\"OperatorParams\":{\"kernel\":\"15\"}"
        "},"

        "{"
        "\"NodeName\":\"P1\","
        "\"NextNodes\":\"FC1\","
        "\"OperatorName\":\"Pooling\","
        "\"OperatorParams\":{}"
        "},"

        "{"
        "\"NodeName\":\"FC1\","
        "\"NextNodes\":\"FC2\","
        "\"OperatorName\":\"FullyConnected\","
        "\"OperatorParams\":{\"kernel\":\"128\"}"
        "},"

        "{"
        "\"NodeName\":\"FC2\","
        "\"NextNodes\":\"LS\","
        "\"OperatorName\":\"FullyConnected\","
        "\"OperatorParams\":{\"kernel\":\"10\"}"
        "},"

        "{"
        "\"NodeName\":\"LS\","
        "\"NextNodes\":\"EndNet\","
        "\"OperatorName\":\"LossFunction\","
        "\"OperatorParams\":{\"loss\":\"softMaxToCrossEntropy\"}"
        "}"

        "],"
        "\"EndNet\":"
        "{"
        "\"PrevNode\":\"LS\""
        "}"
        "}";

    char err[256]{'\0'};
    auto snet = sn::snCreateNet(ss.str().c_str(), err);
    if (!snet){
        cout << "Error 'snCreateNet' " << err << endl;
        system("pause");
        return -1;
    }

    string imgPath = "c:\\cpp\\sunnet\\example\\mnist\\images\\";
  
    int batchSz = 100, classCnt = 10, w = 28, h = 28; float lr = 0.001F;
    vector<vector<string>> imgName(classCnt);
    vector<int> imgCntDir(classCnt);
    map<string, cv::Mat> images;       
   
    if (!loadImage(imgPath, classCnt, imgName, imgCntDir, images)){
        cout << "Error 'loadImage' imgPath: " << imgPath << endl;
        system("pause");
        return -1;
    }
             
    sn::snFloat* inLayer = new sn::snFloat[w * h * batchSz];
    sn::snFloat* targetLayer = new sn::snFloat[classCnt * batchSz];
    sn::snFloat* outLayer = new sn::snFloat[classCnt * batchSz];

    size_t sum_metric = 0;
    size_t num_inst = 0;
    float accuratSumm = 0;
    for (int k = 0; k < 1000; ++k){
               
        srand(clock());

        fill_n(targetLayer, classCnt * batchSz, 0.F);
       
        for (int i = 0; i < batchSz; ++i){

            // directory
            int ndir = rand() % classCnt;
            while (imgCntDir[ndir] == 0) ndir = rand() % classCnt;

            // image
            int nimg = rand() % imgCntDir[ndir];

            // read image
            cv::Mat img; string nm = imgName[ndir][nimg];
            if (images.find(nm) != images.end())
                img = images[nm];
            else{
                img = cv::imread(imgPath + to_string(ndir) + "/" + nm, CV_LOAD_IMAGE_UNCHANGED);
                images[nm] = img;
            }

            float* refData = inLayer + i * w * h;
            double mean = cv::mean(img)[0];
            size_t nr = img.rows, nc = img.cols;
            for (size_t r = 0; r < nr; ++r){
                uchar* pt = img.ptr<uchar>(r);
                for (size_t c = 0; c < nc; ++c)
                    refData[r * nc + c] = pt[c] - mean;
            } 

            float* tarData = targetLayer + i * classCnt;

            tarData[ndir] = 1;
        }

        // training
        float accurat = 0;
        sn::snTraining(snet,
                       lr,
                       sn::snLSize(w, h, 1, batchSz),
                       inLayer,
                       sn::snLSize(10, 1, 1, batchSz),
                       outLayer,
                       targetLayer,                       
                       &accurat);
          
        // calc error
        size_t accCnt = 0;
        for (int i = 0; i < batchSz; ++i){

            float* refTarget = targetLayer + i * classCnt;
            float* refOutput = outLayer + i * classCnt;

            int maxOutInx = distance(refOutput, max_element(refOutput, refOutput + classCnt)),
                maxTargInx = distance(refTarget, max_element(refTarget, refTarget + classCnt));

            if (maxTargInx == maxOutInx)
                ++accCnt;                       
        }

        accuratSumm += (accCnt * 1.F) / batchSz;

        cout << k << " accurate " << accuratSumm / k << endl;
    }

    system("pause");
    return 0;
}