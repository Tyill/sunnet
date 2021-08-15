
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
       
    
    snet.addNode("Input", sn::Input(), "C1 C2")
        .addNode("C1", sn::Convolution(5, 0), "N")
        .addNode("N", sn::Lock(sn::lockType::lock), "P1")
        .addNode("P1", sn::Pooling(), "S")
        .addNode("C2", sn::Convolution(5, 0), "P2")
        .addNode("P2", sn::Pooling(), "S")
        .addNode("S", sn::Summator(), "Output");
      
    string imgPath = "c:/cpp/other/sunnet/example/mnist/images/";

    sn::Tensor inLayer(sn::snLSize(9, 9, 3, 1));
    sn::Tensor outLayer(sn::snLSize(2, 2, 1, 1));

    snet.forward(false, inLayer, outLayer);

        
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
        .addNode("C1", sn::Convolution(15, 0), "C2")
        .addNode("C2", sn::Convolution(15, 0), "P1")
        .addNode("P1", sn::Pooling(sn::calcMode::CUDA), "FC1")
        .addNode("FC1", sn::FullyConnected(128), "FC2")
        .addNode("FC2", sn::FullyConnected(10), "LS")
        .addNode("LS", sn::LossFunction(sn::lossType::softMaxToCrossEntropy), "Output");

    cout << "Hello " <<  SN_API::versionLib() << endl;

}

#endif
