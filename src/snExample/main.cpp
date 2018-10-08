// FNExample.cpp : Defines the entry point for the console application.
//

#include <windows.h>
#include <string>
#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <map>
#include <filesystem>

#include "../cpp/snNet.h"
#include "../cpp/snOperator.h"
#include "snAux/auxFunc.h"
#include "Lib/OpenBLAS/cblas.h"
#include "snBase\snBase.h"

#include "Lib/OpenCV_3.3.0/opencv2/core/core_c.h"
#include "Lib/OpenCV_3.3.0/opencv2/core/core.hpp"
#include "Lib/OpenCV_3.3.0/opencv2/imgproc/imgproc_c.h"
#include "Lib/OpenCV_3.3.0/opencv2/imgproc/imgproc.hpp"
#include "Lib/OpenCV_3.3.0/opencv2/highgui/highgui_c.h"
#include "Lib/OpenCV_3.3.0/opencv2/highgui/highgui.hpp"

using namespace std;
using namespace SN_Base;

mutex mtx;

#define PROFILE_START clock_t ctm = clock(); 
#define PROFILE_END(func) cout << string("Profile ") <<  func <<  " " <<  to_string(clock() - ctm) << endl; ctm = clock(); 

void statusMess(const char* mess, SN_API::snUData){

    mtx.lock();

    cout << SN_Aux::CurrDateTimeMs() << " " << mess << endl;

    mtx.unlock();
}

vector<string> split(string text, const char* sep)
{
    char* cstr = (char*)text.c_str();

    vector<string> res;
    char* pch = strtok(cstr, sep);
    while (pch != NULL){
        res.push_back(string(pch));
        pch = strtok(NULL, sep);
    }

    return res;
}

SN_API::Net createNet(){

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
        "\"OperatorParams\":{\"kernel\":\"15\", \"batchNorm\":\"none\","
        "\"mode\":\"CUDA\","
        "\"padding\":\"0\","
        "\"freeze\":\"0\"}"
        "},"   

        "{"
        "\"NodeName\":\"C2\","
        "\"NextNodes\":\"P1\","
        "\"OperatorName\":\"Convolution\","
        "\"OperatorParams\":{\"kernel\":\"15\", \"batchNorm\":\"none\","
        "\"mode\":\"CUDA\","
        "\"padding\":\"0\","
        "\"freeze\":\"0\"}"
        "},"
        
        "{"
        "\"NodeName\":\"P1\","
        "\"NextNodes\":\"C3\","
        "\"OperatorName\":\"Pooling\","
        "\"OperatorParams\":{"
        "\"mode\":\"CUDA\","
        "\"freeze\":\"0\"}"
        "},"        

        "{"
        "\"NodeName\":\"C3\","
        "\"NextNodes\":\"C4\","
        "\"OperatorName\":\"Convolution\","
        "\"OperatorParams\":{\"kernel\":\"15\", \"batchNorm\":\"none\","
        "\"mode\":\"CUDA\","
        "\"padding\":\"0\","
        "\"freeze\":\"0\"}"
        "},"

        "{"
        "\"NodeName\":\"C4\","
        "\"NextNodes\":\"P2\","
        "\"OperatorName\":\"Convolution\","
        "\"OperatorParams\":{\"kernel\":\"15\", \"batchNorm\":\"none\","
        "\"mode\":\"CUDA\","
        "\"padding\":\"0\","
        "\"freeze\":\"0\"}"
        "},"
		
		
        "{"
        "\"NodeName\":\"P2\","
        "\"NextNodes\":\"U1\","
        "\"OperatorName\":\"Pooling\","
        "\"OperatorParams\":{"
        "\"mode\":\"CUDA\","
        "\"freeze\":\"0\"}"
        "},"

		
		
        "{"
        "\"NodeName\":\"U1\","
        "\"NextNodes\":\"U2\","
        "\"OperatorName\":\"Deconvolution\","
        "\"OperatorParams\":{\"kernel\":\"15\", \"batchNorm\":\"none\","
        "\"mode\":\"CUDA\","
        "\"gpuClearMem\":\"0\","
        "\"stride\":\"2\","
        "\"freeze\":\"0\"}"
        "},"

        "{"
        "\"NodeName\":\"U2\","
        "\"NextNodes\":\"LS\","
        "\"OperatorName\":\"Deconvolution\","
        "\"OperatorParams\":{\"kernel\":\"1\", \"batchNorm\":\"none\","
        "\"mode\":\"CUDA\","
        "\"stride\":\"2\","
        "\"active\":\"sigmoid\","
        "\"freeze\":\"0\"}"
        "},"

        "{"
        "\"NodeName\":\"LS\","
        "\"NextNodes\":\"EndNet\","
        "\"OperatorName\":\"LossFunction\","
        //"\"OperatorParams\":{\"loss\":\"softMaxToCrossEntropy\"}"
        "\"OperatorParams\":{\"loss\":\"binaryCrossEntropy\"}"
        "}"

        "],"
        "\"EndNet\":"                              ///< выход ветви
        "{"
        "\"PrevNode\":\"LS\""               ///< Предыд узел ветви.
        "}"
        "}";

                   
    return SN_API::Net(ss.str().c_str());
    
    //std::ifstream ifs;
    //ifs.open("c:\\C++\\VTD\\s20302_Isolation\\res\\cnn\\struct.txt", std::ifstream::in);

    //if (!ifs.good()){
    //    statusMess("isolFinder::createNet error open file: ", nullptr);
    //    return false;
    //}
    //int tt = 0;
    //ifs >> tt;
    //ifs >> tt;

    //int cp = ifs.tellg();
    //ifs.seekg(0, ifs.end);
    //size_t length = ifs.tellg();
    //ifs.seekg(cp, ifs.beg);

    //string jnNet; jnNet.resize(length);
    //ifs.read((char*)jnNet.data(), length);

    //// создаем сеть
    //char err[256]; err[0] = '\0';
    //net = SN_API::snCreateNet(jnNet.c_str(), err, statusMess);

}

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

int main(int argc, char* argv[])
{
    namespace sn = SN_API;

    auto ver = sn::versionLib();

    sn::Net snet;
    
    auto c1 = sn::Convolution(15, sn::calcMode::CUDA);
    c1.padding = 0;
    c1.gpuClearMem = false;
    c1.freeze = false;
    c1.bnorm = sn::batchNormType::none;

    auto dc1 = sn::Deconvolution(15, sn::calcMode::CUDA);
    dc1.gpuClearMem = true;
    dc1.freeze = false;
    dc1.bnorm = sn::batchNormType::beforeActive;

   // auto p1 = sn::Pooling(2, sn::poolType::avg, sn::calcMode::CUDA);
   // p1.gpuClearMem = false;
  
    auto fc1 = sn::FullyConnected(10, sn::calcMode::CUDA);
   // fc1.freeze = true;
    fc1.gpuClearMem = true;

    snet.addNode("Input", sn::Input(), "C1")
        .addNode("C1", c1, "DC1")
        .addNode("DC1", dc1, "FC1")
        .addNode("FC1", fc1, "LS")
        .addNode("LS", sn::LossFunction(sn::lossType::softMaxToCrossEntropy), "Output");


  /*  if (!snet.getLastErrorStr().empty()){
        statusMess("createNet error", nullptr);
        cin.get();
        return -1;
    }*/

    
  // SN_API::snLoadAllWeightFromFile(snet, "c:\\\C++\\ww\\w.dat");

   // string imgPath = "c:\\C++\\skyNet\\test\\unet\\membrane\\train\\";
   // string imgTargPath = "c:\\C++\\skyNet\\test\\unet\\membrane\\train\\label\\";
    string imgPath = "d:\\Работа\\CNN\\Mnist/training/";
    //string imgPath = "d:\\Работа\\CNN\\ТипИзоляции\\ОбучВыборка2\\";

    int batchSz = 100, classCnt = 10, w = 28, h = 28, w1 = 10, h1 = 1, d = 1; float lr = 0.01F; //28
    vector<vector<string>> imgName(classCnt);
    vector<int> imgCntDir(classCnt);
    map<string, cv::Mat> images;
       
    if (!loadImage(imgPath, classCnt, imgName, imgCntDir, images)){
        statusMess("loadImage error", nullptr);
        cin.get();
        return -1;
    }

    sn::Tensor inLayer(sn::snLSize(w, h, d, batchSz));
    sn::Tensor targetLayer(sn::snLSize(w1, h1, d, batchSz));
    sn::Tensor outLayer(sn::snLSize(w1, h1, d, batchSz));
       
    size_t sum_metric = 0;
    size_t num_inst = 0;
    float accuratSumm = 0;
    for (int k = 0; k < 10; ++k){

        targetLayer.clear();
        outLayer.clear();

        Sleep(1); srand(clock());

        // выбираем случайно изобр из тренир выборки

        for (int i = 0; i < d * batchSz; ++i){

            // директория
            int ndir = rand() % classCnt;
            while (imgCntDir[ndir] == 0) ndir = rand() % classCnt;

            // номер изобр
            int nimg = rand() % imgCntDir[ndir];

            // читаем изобр
            cv::Mat img; string nm = imgName[ndir][nimg];
            if (images.find(nm) != images.end())
                img = images[nm];
            else{
                img = cv::imread(imgPath + to_string(ndir) + "/" + nm, CV_LOAD_IMAGE_UNCHANGED);
                images[nm] = img;
            }

            cv::resize(img, img, cv::Size(w, h));

            float* refData = inLayer.data() + i * w * h;
           
            double mean = cv::mean(img)[0];
            size_t nr = img.rows, nc = img.cols;
            for (size_t r = 0; r < nr; ++r){
                uchar* pt = img.ptr<uchar>(r);
                for (size_t c = 0; c < nc; ++c)
                    refData[r * nc + c] = pt[c];
            } 

            float* tarData = targetLayer.data() + i * w1 * h1;

            tarData[ndir] = 1;
        }

        float accurat = 0;
        snet.training(lr,
            inLayer,
            outLayer,
            targetLayer,
            accurat);

        snFloat* targetData = targetLayer.data();
        snFloat* outData = outLayer.data();

        size_t accCnt = 0, bsz = batchSz, osz = w1;
        for (int i = 0; i < bsz; ++i){

            float* refTarget = targetData + i * osz;
            float* refOutput = outData + i * osz;

            if (osz > 1){
                int maxOutInx = distance(refOutput, max_element(refOutput, refOutput + osz)),
                    maxTargInx = distance(refTarget, max_element(refTarget, refTarget + osz));

                if (maxTargInx == maxOutInx)
                    ++accCnt;
            }
            else{

                if (abs(refOutput[0] - refTarget[0]) < 0.1)
                    ++accCnt;
            }
        }

        accurat = (accCnt * 1.F) / bsz;


        accuratSumm += accurat;     
        cout << k << " metrix " << accuratSumm / k << " " << snet.getLastErrorStr() << endl;

        
    }

    snet.~Net();

    //SN_API::Tensor ow;
   // snet.getWeightNode("FC2", ow);

    

    //SN_API::snSaveAllWeightToFile(snet, "c:\\\C++\\ww\\w.dat");
    //writeWeight(snet);


    cin.get();

    return 0;
}
