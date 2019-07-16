
#ifndef CV_VERSION

#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <map>

#ifdef WIN32
#include <filesystem>
namespace fs = std::tr2::sys;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif // WIN32


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
    
    bool ok = false;
    for (int i = 0; i < classCnt; ++i){

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

        if (cnt > 0)
            ok = true;

        imgCntDir[i] = cnt;
    }

    return ok;
}


int main(int argc, char* argv[]){
       
    sn::Net snet;

    sn::Convolution cnv(1, 7, 0, 2);

    cnv.act = sn::active::none;
    cnv.useBias = false;
   
    snet.addNode("Input", sn::Input(), "C1")
        .addNode("C1", cnv, "Output");
      
#ifdef WIN32
     string imgPath = "c:/cpp/other/skyNet/example/mnist/images/";
#else
     string imgPath = "/home/alex/CLionProjects/skynet/example/mnist/images/";
#endif

    sn::Tensor inLayer(sn::snLSize(9, 9, 3, 1));
    sn::Tensor outLayer(sn::snLSize(2, 2, 1, 1));

    float* refData = inLayer.data();
   /*srand(clock());
    _sleep(100);
     srand(clock());
    string ff;
    for (int i = 0; i < 49; ++i)
     ff += to_string(rand() % 100) + ", "*/;

       
    float b1[81]{ 41, 67, 34, 0, 69, 24, 78, 58, 62,
        64, 5, 45, 81, 27, 61, 91, 95, 42,
        27, 36, 91, 4, 2, 53, 92, 82, 21,
        16, 18, 95, 47, 26, 71, 38, 69, 12,
        67, 99, 35, 94, 3, 11, 22, 33, 73,
        64, 41, 11, 53, 68, 47, 44, 62, 57,
        37, 59, 23, 41, 29, 78, 16, 35, 90,
        42, 88, 6, 40, 42, 64, 48, 46, 5,
        90, 29, 70, 50, 6, 1, 93, 48, 29};

    float b2[81]{ 57, 80, 25, 16, 72, 55, 25, 1, 53,
        41, 4, 36, 48, 7, 47, 65, 65, 75,
        37, 50, 33, 92, 49, 15, 2, 65, 68,
        34, 42, 71, 44, 96, 19, 24, 41, 63,
        95, 42, 23, 41, 41, 0, 73, 61, 94,
        19, 57, 89, 6, 24, 15, 20, 92, 76,
        10, 36, 7, 0, 0, 54, 81, 6, 41,
        16, 56, 90, 84, 36, 32, 26, 41, 8,
        24, 54, 83, 13, 32, 31, 24, 72, 30};

    float b3[81]{ 24, 54, 88, 58, 62, 44, 35, 71, 39,
        52, 90, 11, 74, 17, 38, 52, 42, 47,
        44, 98, 0, 68, 32, 22, 27, 81, 52,
        50, 68, 79, 60, 36, 77, 16, 13, 81,
        17, 14, 16, 88, 29, 70, 4, 63, 81,
        41, 56, 29, 70, 7, 62, 92, 47, 99,
        58, 85, 33, 47, 87, 18, 32, 56, 3,
        99, 22, 37, 90, 8, 6, 4, 17, 10,
        90, 25, 86, 64, 88, 62, 11, 16, 27};
       
    memcpy(refData, b1, 81 * sizeof(float));
    memcpy(refData + 81, b2, 81 * sizeof(float));
    memcpy(refData + 162, b3, 81 * sizeof(float));

       
    sn::Tensor weight(sn::snLSize(7, 7, 3, 1));
      
    float w1[49] { 41, 67, 34, 0, 69, 24, 78,
        58, 62, 64, 5, 45, 81, 27,
        61, 91, 95, 42, 27, 36, 91,
        4, 2, 53, 92, 82, 21, 16,
        18, 95, 47, 26, 71, 38, 69,
        12, 67, 99, 35, 94, 3, 11,
        22, 33, 73, 64, 41, 11, 53};

    float w2[49] { 28, 35, 84, 53, 76, 71, 47,
        43, 50, 4, 43, 55, 45, 96,
        31, 22, 92, 4, 74, 55, 15,
        81, 79, 69, 62, 40, 93, 58,
        61, 96, 61, 5, 8, 67, 43,
        57, 73, 95, 95, 47, 0, 34,
        4, 10, 58, 20, 85, 97, 45};

    float w3[49] { 57, 80, 25, 16, 72, 55, 25,
        1, 53, 41, 4, 36, 48, 7,
        47, 65, 65, 75, 37, 50, 33,
        92, 49, 15, 2, 65, 68, 34,
        42, 71, 44, 96, 19, 24, 41,
        63, 95, 42, 23, 41, 41, 0,
        73, 61, 94, 19, 57, 89, 6};
                          
          
    float* refWeight = weight.data();

    memcpy(refWeight, w1, 49 * sizeof(float));
    memcpy(refWeight + 49, w2, 49 * sizeof(float));
    memcpy(refWeight + 98, w3, 49 * sizeof(float));
   
    snet.setWeightNode("C1", weight);

    snet.forward(false, inLayer, outLayer);

    for (int i = 0; i < outLayer.size().w * outLayer.size().h * outLayer.size().ch; ++i)
        cout << outLayer.data()[i] << " ";

    cout << endl << "get last error: " << snet.getLastErrorStr() << endl;
        
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
