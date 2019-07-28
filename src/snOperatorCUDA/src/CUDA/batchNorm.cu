
#include <cuda_runtime.h>

#include "../stdafx.h"
#include "../cudaCommon.h"

using namespace SN_Base;

__global__ void batchNormInf(snSize insz, snFloat* in, snFloat* out, batchNorm prm){
    
    size_t inStepByD = insz.w * insz.h,     // step out by input
           inStepByN = inStepByD * insz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size
        
    snFloat* cin = in + inStepByN * blockIdx.y + inStepByD * blockIdx.x,
           * cout = out + inStepByN * blockIdx.y + inStepByD * blockIdx.x;
    
    prm.mean += inStepByD * blockIdx.x;
    prm.varce += inStepByD * blockIdx.x;
    prm.scale += inStepByD * blockIdx.x;
    prm.schift += inStepByD * blockIdx.x;
    
    unsigned int i = threadIdx.x;
    while (i < inStepByD){
                      
        cout[i] = (cin[i] - prm.mean[i]) * prm.scale[i] / prm.varce[i] + prm.schift[i];

        i += blockDim.x;
    }
}

__global__ void calcMeanAndVarce(SN_Base::snSize insz, snFloat* in, batchNorm prm){

    size_t sz = insz.w * insz.h * insz.d,
           bsz = insz.n;

    // gridDim.x = insz.d    
    // blockDim.x <= insz.w * insz.h

    in += insz.w * insz.h * blockIdx.x;
    prm.mean += insz.w * insz.h * blockIdx.x;
    prm.varce += insz.w * insz.h * blockIdx.x;

    unsigned int i = threadIdx.x;
    while (i < (insz.w * insz.h)){
               
        snFloat srq = 0.F, sum = 0.F;
        for (size_t j = 0; j < bsz; ++j){
           
            snFloat cin = in[i + j * sz];

            sum += cin;

            srq += cin * cin;
        }
               
        prm.mean[i] = sum / bsz;
        srq /= bsz;

        prm.varce[i] = sqrt(srq - prm.mean[i] * prm.mean[i] + 0.00001F);
                       
        i += blockDim.x;
    }
}

__global__ void calcNormAndOut(SN_Base::snSize insz, snFloat* in, snFloat* out, batchNorm prm){
       
    size_t sz = insz.w * insz.h * insz.d;

    // gridDim.x = insz.d
    // gridDim.y = insz.n
    // blockDim.x <= insz.w * insz.h
    
    prm.mean += blockIdx.x * insz.w * insz.h;
    prm.varce += blockIdx.x * insz.w * insz.h;
    prm.scale += blockIdx.x * insz.w * insz.h;
    prm.schift += blockIdx.x * insz.w * insz.h;

    snFloat* cin = in + blockIdx.x * insz.w * insz.h + blockIdx.y * sz,
           * cout = out + blockIdx.x * insz.w * insz.h + blockIdx.y * sz,
           * norm = prm.norm + blockIdx.x * insz.w * insz.h + blockIdx.y * sz;

    unsigned int i = threadIdx.x;
    while (i < (insz.w * insz.h)){
               
        norm[i] = (cin[i] - prm.mean[i]) / prm.varce[i];
        cout[i] = norm[i] * prm.scale[i] + prm.schift[i];
       
        i += blockDim.x;
    }    
}

__global__ void calcDSchiftAndDScale(SN_Base::snSize insz, snFloat* gradIn, batchNorm prm){

    size_t sz = insz.w * insz.h * insz.d,
           bsz = insz.n;

    // gridDim.x <= insz.d
    // blockDim.x <= insz.w * insz.h

    gradIn += insz.w * insz.h * blockIdx.x;
    prm.norm += insz.w * insz.h * blockIdx.x;
    prm.dSchift += insz.w * insz.h * blockIdx.x;
    prm.dScale += insz.w * insz.h * blockIdx.x;

    unsigned int i = threadIdx.x;
    while (i < (insz.w * insz.h)){
                
        snFloat dScale = 0.F, sum = 0.F;
        for (size_t j = 0; j < bsz; ++j){
            
            snFloat cin = gradIn[i + j * sz],
                    norm = prm.norm[i + j * sz];
          
            sum += cin;
            
            dScale += cin * norm;
        }
        prm.dSchift[i] = sum;
        prm.dScale[i] = dScale;
              
        i += blockDim.x;
    }
}

__global__ void calcGrOut(SN_Base::snSize insz, snFloat* gradIn, snFloat* gradOut, batchNorm prm){

    size_t sz = insz.w * insz.h * insz.d,
           bsz = insz.n;

    // gridDim.x = insz.d
    // gridDim.y = insz.n
    // blockDim.x <= insz.w * insz.h

    prm.scale += insz.w * insz.h * blockIdx.x;
    prm.varce += insz.w * insz.h * blockIdx.x;
    prm.dSchift += insz.w * insz.h * blockIdx.x;
    prm.dScale += insz.w * insz.h * blockIdx.x;

    snFloat* igr = gradIn + blockIdx.x * insz.w * insz.h + blockIdx.y * sz,
           * ogr = gradOut + blockIdx.x * insz.w * insz.h + blockIdx.y * sz,
           * norm = prm.norm + blockIdx.x * insz.w * insz.h + blockIdx.y * sz;
       
    unsigned int i = threadIdx.x;
    while (i < (insz.w * insz.h)){

        ogr[i] = prm.scale[i] * (igr[i] * bsz - prm.dSchift[i] - norm[i] * prm.dScale[i]) / (prm.varce[i] * bsz);

        i += blockDim.x;
    }
}

__global__ void calcSchiftAndScale(SN_Base::snSize insz, batchNorm prm){
       
    // gridDim.x = insz.d
    // blockDim.x <= insz.w * insz.h

    prm.schift += insz.w * insz.h * blockIdx.x;
    prm.scale += insz.w * insz.h * blockIdx.x;
    prm.dSchift += insz.w * insz.h * blockIdx.x;
    prm.dScale += insz.w * insz.h * blockIdx.x;

    unsigned int i = threadIdx.x;
    while (i < (insz.w * insz.h)){
              
        prm.schift[i] -= prm.dSchift[i] * prm.lr;
        prm.scale[i] -= prm.dScale[i] * prm.lr;

        i += blockDim.x;
    }
}


void batchNormForward(bool isLern, const snSize& insz, snFloat* in, snFloat* out, const batchNorm& prm){
        
    if (!isLern){

        dim3 dimBlock(128);
        dim3 dimGrid(int(insz.d), int(insz.n));

        batchNormInf << <dimGrid, dimBlock >> > (insz, in, out, prm);
    }
    else{
     
        calcMeanAndVarce << < int(insz.d), 256 >> > (insz, in, prm);

        dim3 dimBlock(128);
        dim3 dimGrid(int(insz.d), int(insz.n));

        calcNormAndOut << < dimGrid, dimBlock >> > (insz, in, out, prm);            
    }
}

void batchNormBackward(const snSize& insz, snFloat* in, snFloat* out, const batchNorm& prm){
      
    calcDSchiftAndDScale << <int(insz.d), 256 >> > (insz, in, prm);
        
    dim3 dimBlock(128);
    dim3 dimGrid(int(insz.d), int(insz.n));

    calcGrOut << < dimGrid, dimBlock >> > (insz, in, out, prm);       

    calcSchiftAndScale << <int(insz.d), 256 >> > (insz, prm);   
}

