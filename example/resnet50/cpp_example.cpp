
#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <map>
#include <filesystem>


#include <omp.h>
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


void idntBlock(sn::Net& net, vector<uint32_t>&& filters, uint32_t kernelSize, string oprName, string nextOprName){
    
    net.addNode(oprName + "2a", sn::Convolution(filters[0], 1, 0, 1, sn::batchNormType::beforeActive, sn::active::relu), oprName + "2b")
       .addNode(oprName + "2b", sn::Convolution(filters[1], kernelSize, -1, 1, sn::batchNormType::beforeActive, sn::active::relu), oprName + "2c")
       .addNode(oprName + "2c", sn::Convolution(filters[2], 1, 0, 1, sn::batchNormType::beforeActive, sn::active::none), oprName + "Sum");

    net.addNode(oprName + "Sum", sn::Summator(sn::summatorType::summ), oprName + "Act")
       .addNode(oprName + "Act", sn::Activation(sn::active::relu), nextOprName);
}

void convBlock(sn::Net& net, vector<uint32_t>&& filters, uint32_t kernelSize, uint32_t stride, string oprName, string nextOprName){

    net.addNode(oprName + "2a", sn::Convolution(filters[0], 1, 0, stride, sn::batchNormType::beforeActive, sn::active::relu), oprName + "2b")
       .addNode(oprName + "2b", sn::Convolution(filters[1], kernelSize, -1, 1, sn::batchNormType::beforeActive, sn::active::relu), oprName + "2c")
       .addNode(oprName + "2c", sn::Convolution(filters[2], 1, 0, 1, sn::batchNormType::beforeActive, sn::active::none), oprName + "Sum");

    // shortcut
    net.addNode(oprName + "1", sn::Convolution(filters[2], 1, 0, stride, sn::batchNormType::beforeActive, sn::active::none), oprName + "Sum");

    // summator
    net.addNode(oprName + "Sum", sn::Summator(sn::summatorType::summ), oprName + "Act")
       .addNode(oprName + "Act", sn::Activation(sn::active::relu), nextOprName);
}

sn::Net createNet(){

    auto net = sn::Net();

    net.addNode("In", sn::Input(), "conv1")
       .addNode("conv1", sn::Convolution(64, 7, 3, 2, sn::batchNormType::beforeActive, sn::active::none), "pool1_pad")
       .addNode("pool1_pad", sn::Pooling(3, 2, sn::poolType::max), "res2a_branch1 res2a_branch2a");
    
    convBlock(net, vector<uint32_t>{ 64, 64, 256 }, 3, 1, "res2a_branch", "res2b_branch2a res2b_branchSum");
    idntBlock(net, vector<uint32_t>{ 64, 64, 256 }, 3, "res2b_branch", "res2c_branch2a res2c_branchSum");
    idntBlock(net, vector<uint32_t>{ 64, 64, 256}, 3, "res2c_branch", "res3a_branch1 res3a_branch2a");

    convBlock(net, vector<uint32_t>{ 128, 128, 512 }, 3, 2, "res3a_branch", "res3b_branch2a res3b_branchSum");
    idntBlock(net, vector<uint32_t>{ 128, 128, 512 }, 3, "res3b_branch", "res3c_branch2a res3c_branchSum");
    idntBlock(net, vector<uint32_t>{ 128, 128, 512 }, 3, "res3c_branch", "res3d_branch2a res3d_branchSum");
    idntBlock(net, vector<uint32_t>{ 128, 128, 512 }, 3, "res3d_branch", "res4a_branch1 res4a_branch2a");

    convBlock(net, vector<uint32_t>{ 256, 256, 1024 }, 3, 2, "res4a_branch", "res4b_branch2a res4b_branchSum");
    idntBlock(net, vector<uint32_t>{ 256, 256, 1024 }, 3, "res4b_branch", "res4c_branch2a res4c_branchSum");
    idntBlock(net, vector<uint32_t>{ 256, 256, 1024 }, 3, "res4c_branch", "res4d_branch2a res4d_branchSum");
    idntBlock(net, vector<uint32_t>{ 256, 256, 1024 }, 3, "res4d_branch", "res4e_branch2a res4e_branchSum");
    idntBlock(net, vector<uint32_t>{ 256, 256, 1024 }, 3, "res4e_branch", "res4f_branch2a res4f_branchSum");
    idntBlock(net, vector<uint32_t>{ 256, 256, 1024 }, 3, "res4f_branch", "res5a_branch1 res5a_branch2a");

    convBlock(net, vector<uint32_t>{ 512, 512, 2048 }, 3, 2, "res5a_branch", "res5b_branch2a res5b_branchSum");
    idntBlock(net, vector<uint32_t>{ 512, 512, 2048 }, 3, "res5b_branch", "res5c_branch2a res5c_branchSum");
    idntBlock(net, vector<uint32_t>{ 512, 512, 2048 }, 3, "res5c_branch", "avg_pool");

    net.addNode("avg_pool", sn::Pooling(7, 7, sn::poolType::avg), "fc1000")
       .addNode("fc1000", sn::FullyConnected(1000, sn::active::none), "LS")
       .addNode("LS", sn::LossFunction(sn::lossType::softMaxToCrossEntropy), "Output");

    return net;
}

int main(int argc, char* argv[]){
       
    sn::Net snet = createNet();
  
    // using python for create file 'resNet50Weights.dat' as: 
    // CMD: cd c:\cpp\other\skyNet\example\resnet50\
    // CMD: python createNet.py  
    
    if (!snet.loadAllWeightFromFile("c:/cpp/other/skyNet/example/resnet50/resNet50Weights.dat")){
        cout << "error loadAllWeightFromFile: " << snet.getLastErrorStr() << endl;
        system("pause");
        return -1;
    }
    
    string imgPath = "c:/cpp/other/skyNet/example/resnet50/images/elephant.jpg";
    
    int classCnt = 1000, w = 224, h = 224;
  
    sn::Tensor inLayer(sn::snLSize(w, h, 3, 1));
    sn::Tensor outLayer(sn::snLSize(classCnt, 1, 1, 1));
  
    cv::Mat img = cv::imread(imgPath, CV_LOAD_IMAGE_UNCHANGED);

    if ((img.cols != w) || (img.rows != h))
       cv::resize(img, img, cv::Size(w, h));

    vector<cv::Mat> channels(3);
    split(img, channels);    // BGR order 

    float* refData = inLayer.data();
      
    // B
    for (size_t r = 0; r < h; ++r){
        uchar* pt = channels[0].ptr<uchar>(r);
        for (size_t c = 0; c < w; ++c)
            refData[r * w + c] = pt[c];
    } 

    // G
    refData += h * w;
    for (size_t r = 0; r < h; ++r){
        uchar* pt = channels[1].ptr<uchar>(r);
        for (size_t c = 0; c < w; ++c)
            refData[r * w + c] = pt[c];
    }

    // R
    refData += h * w;   
    for (size_t r = 0; r < h; ++r){
        uchar* pt = channels[2].ptr<uchar>(r);
        for (size_t c = 0; c < w; ++c)
            refData[r * w + c] = pt[c];
    }


    for (int i = 0; i < 100; ++i){
        double ctm = omp_get_wtime();        
        snet.forward(false, inLayer, outLayer);
        cout << std::to_string(omp_get_wtime() - ctm) << endl;
    
        float* refOutput = outLayer.data();

        int maxInx = distance(refOutput, max_element(refOutput, refOutput + classCnt));

        // for check: c:\cpp\other\skyNet\example\resnet50\imagenet_class_index.json
    
        cout << "inx " << maxInx << " accurate " << refOutput[maxInx] << snet.getLastErrorStr() << endl;
    }

    system("pause");
    return 0;
}
