// FNExample.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <windows.h>
#include <string>
#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <map>
#include <filesystem>

#include "skynet/skynet.h"
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


int main(int argc, _TCHAR* argv[])
{
    stringstream ss;

    ss << "{"
        
        "\"BeginNet\":"                         
        "{"
        "\"NextNodes\":\"F1\""                     
        "},"

        "\"Nodes\":"                            
        "["

        "{"
        "\"NodeName\":\"F1\","    
        "\"NextNodes\":\"F2\","   
        "\"OperatorName\":\"Convolution\","  
        "\"OperatorParams\":{\"kernel\":\"32\"," 
                            "\"krnWidth\":\"3\","
                            "\"krnHeight\":\"3\","
                            "\"padding\":\"1\","
                            "\"stride\":\"1\","
                            "\"weightInitType\":\"he\","
                            "\"activeType\":\"relu\","  
                            "\"optimizerType\":\"adam\"," 
                            "\"batchNormType\":\"none\"}"    
        "},"
       
       "{"
        "\"NodeName\":\"F2\","
        "\"NextNodes\":\"F3\","
        "\"OperatorName\":\"Pooling\","
        "\"OperatorParams\":{\"kernel\":\"2\"}"
        "},"   
        
        "{"
        "\"NodeName\":\"F3\","
        "\"NextNodes\":\"F4\","
        "\"OperatorName\":\"Convolution\","
        "\"OperatorParams\":{\"kernel\":\"64\","
        "\"krnWidth\":\"3\","
        "\"krnHeight\":\"3\","
        "\"padding\":\"same\","
        "\"stride\":\"1\","
        "\"weightInitType\":\"he\","
        "\"activeType\":\"relu\","
        "\"optimizerType\":\"adam\","
        "\"batchNormType\":\"none\"}"
        "},"

        "{"
        "\"NodeName\":\"F4\","
        "\"NextNodes\":\"F5\","
        "\"OperatorName\":\"Pooling\","
        "\"OperatorParams\":{\"kernel\":\"2\"}"
        "},"
        
        "{"
        "\"NodeName\":\"F5\","
        "\"NextNodes\":\"F6\","
        "\"OperatorName\":\"FullyConnected\","
        "\"OperatorParams\":{\"kernel\":\"1024\","
                            "\"weightInitType\":\"he\","
                            "\"activeType\":\"relu\","
                            "\"optimizerType\":\"adam\","
                            "\"batchNormType\":\"none\"}"
        "},"

        "{"
        "\"NodeName\":\"F6\","
        "\"NextNodes\":\"LS\","
        "\"OperatorName\":\"FullyConnected\","
        "\"OperatorParams\":{\"kernel\":\"10\","
        "\"weightInitType\":\"uniform\","
        "\"activeType\":\"none\","
        "\"optimizerType\":\"adam\","
        "\"batchNormType\":\"none\"}"
        "},"
                
        "{"
        "\"NodeName\":\"LS\","    
        "\"NextNodes\":\"EndNet\","  
        "\"OperatorName\":\"LossFunction\","  
        "\"OperatorParams\":{\"lossType\":\"softMaxToCrossEntropy\"}" 
    //    "\"OperatorParams\":{\"lossType\":\"binaryCrossEntropy\" 
        "}"

        "],"
        "\"EndNet\":"                              ///< выход ветви
        "{"
        "\"PrevNode\":\"LS\""               ///< Предыд узел ветви.
        "}"
        "}";


    char err[256];
    auto snet = SN_API::snCreateNet(ss.str().c_str(), err, statusMess);
    string imgPath = "d:\\Работа\\CNN\\Mnist/training/";
    //string imgPath = "d:\\Работа\\CNN\\ТипИзоляции\\ОбучВыборка2\\";
        
    int batchSz = 24, classCnt = 10, w = 28, h = 28;
    SN_API::snFloat* inLayer = new SN_API::snFloat[w * h * batchSz];
    SN_API::snFloat* targetLayer = new SN_API::snFloat[classCnt * batchSz];
    SN_API::snFloat* outLayer = new SN_API::snFloat[classCnt * batchSz];
    
    // собираем все изобр
    vector<vector<string>> imgName(classCnt);
    vector<int> imgCntDir(classCnt);
    map<string, cv::Mat> images;
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

    
    /*std::ifstream ifs;
    ifs.open("c:/C++/w.dat", std::ifstream::in | std::ifstream::binary);

    if (!ifs.good()) return false;

    string str = "";
    getline(ifs, str);

    auto insz = split(str, " ");*/
    /*if (insz.size() != 3 || !is_number(insz[1]) || !is_number(insz[2])){
        statusMess("isolFinder::createNet error: input size not set");
        return false;
    }*/

    /*size_t dbSz = stoi(insz[1]) * stoi(insz[2]);

    std::vector<SN_API::snFloat> db(dbSz);
    if (dbSz > 0){
        ifs.read((char*)db.data(), dbSz * sizeof(SN_API::snFloat));
    }

    ifs.close();

    SN_API::snLSize lsz(stoi(insz[1]), stoi(insz[2]));*/

    //SN_API::snSetWeightNode(snet, "F1", db.data(), lsz);


    size_t sum_metric = 0;
    size_t num_inst = 0;

fff:
    float accuratSumm = 0, lr = 0.0001;
    for (int k = 0; k < 10000; ++k){

        fill_n(targetLayer, classCnt * batchSz, 0.F);
        fill_n(outLayer, classCnt * batchSz, 0.F);

        Sleep(1); srand(clock());

        // выбираем случайно изобр из тренир выборки

        for (int i = 0; i < batchSz; ++i){
            
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
            
            float* refData = inLayer + i * w * h;

            float* refTarget = targetLayer + i * classCnt;
            refTarget[ndir] = 1.F;

        //    refTarget[0] = (ndir == 0) ? 1.F : 0.F;

            /*cv::Mat imggg = img.clone();
            cv::resize(imggg, imggg, cv::Size(500, 500));
            string tt;
            for (int h = 0; h < 10; ++h)
                tt += " " + to_string(int(refTarget[h])) + " ";
            cv::putText(imggg, tt,
                cv::Point(25, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar::all(255), 1);
            
            cv::imshow("", imggg);
            cv::waitKey(1000);            
*/
            double minVal = 0, maxVal = 0;
            double mean = cv::mean(img)[0];// minv = 0, maxv = 0;
            cv::minMaxLoc(img, &minVal, &maxVal);
            size_t nr = img.rows, nc = img.cols;
            for (size_t r = 0; r < nr; ++r){

                uchar* pt = img.ptr<uchar>(r);

                for (size_t c = 0; c < nc; ++c){
                                        
                    refData[r * nc + c] = (pt[c] - mean);
                }
            }

        }

        /*if (k % 100 == 0)
            lr -= lr * 0.1;*/

        float accurat = 0;
        SN_API::snTraining(snet,
                           lr,
                           inLayer,
                           SN_API::snLSize(w, h, 1, batchSz),
                           targetLayer,
                           outLayer,
                           SN_API::snLSize(10, 1, 1, batchSz),
                           &accurat);

        accuratSumm += accurat;
        cout << k << " metrix " << accuratSumm / k << endl;
    }
    
    /*SN_API::batchNorm bn;
    SN_API::snLSize lsz;
    SN_API::snGetBatchNormNode(snet, "F1", &bn, &lsz);

    SN_API::snFloat* weight = nullptr;
    SN_API::snLSize wsz;
    SN_API::snGetWeightNode(snet, "F1", &weight, &wsz);

    
    snet = SN_API::snCreateNet(ss.str().c_str(), err, "", statusMess);

    SN_API::snSetBatchNormNode(snet, "F1", bn, lsz);

    SN_API::snSetWeightNode(snet, "F1", weight, wsz);*/


    //goto fff;

    cin.get();



    /*std::ofstream ofs;        
    ofs.open("c:/C++/w.dat", std::ios_base::out | std::ios_base::binary);

    if (ofs.good()) {
            
        SN_API::snFloat* data = nullptr;
        SN_API::snLSize lSize;
        
        SN_API::snGetWeightNode(snet, "F1", &data, &lSize);
            
        ofs << "F1 " << lSize.w << " " << lSize.h << endl;
        ofs.write((char*)data, lSize.w * lSize.h * sizeof(float));
    }
*/

//
//
//float a[15] = { 3., 1., 2., 3., 3.,                  // 3   0  5   
//               0., -1., 4., 3., 8., // 2 x 4 matrix  // 1  -1  6
//               5., 6., 7., 3., 8. };                 // 2   4  7
//                                                     // 3   3  3 
//float b[5] = { 1., 1., 1.};                          // 3   8  8 
//
//float c[25] = { 0., 0., 0., 0., 0.,
//                0., 0., 0., 0., 0.,
//                0., 0., 0., 0., 0.,
//                0., 0., 0., 0., 0.,
//                0., 0., 0., 0., 0. };
//
//rmsOfBatch(false, snSize(5, 1, 1, 3), a, c);
//return 0;
//cblas_sgemm(CBLAS_ORDER::CblasRowMajor,
//    CBLAS_TRANSPOSE::CblasNoTrans,
//    CBLAS_TRANSPOSE::CblasTrans,
//    3,                        // Матрица А M - строк, кол-во изобр в батче
//    2,                        // Матрица В N - столбцов, кол-во скрытых нейронов 
//    5,                        // Матрица A N - столбцов, В М - строк, кол-во вх нейронов - размер одного изображения ((w + 1) * h * d) из батча. (+1 - X0)                   
//    1.0,                      // α - доп коэф умн на результат AB
//    a,                        // Матрица А - данные
//    5,                        // Матрица А - до след а
//    b,                        // Матрица B - данные
//    5,                        // Матрица B - до след в
//    0.0,                      // β - доп коэф умн на результат C
//    c,                        // Матрица С - выходные данные
//    2);                       // Матрица С - до след c
//
//
//



    return 0;    
}

