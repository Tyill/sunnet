
#include <cuda_runtime.h>

#include "../stdafx.h"

using namespace SN_Base;

__device__ void bnOffset(batchNorm* bn, size_t offs){
    bn->mean += offs;
    bn->varce += offs;
    bn->scale += offs;
    bn->dScale += offs;
    bn->schift += offs;
    bn->dSchift += offs;
}

__global__ void channelBatchNormInf(snSize insz, snFloat* in, snFloat* out, batchNorm prm){
    
    size_t inStepByD = insz.w * insz.h,     // step out by input
           inStepByN = inStepByD * insz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size
        
    snFloat* cin = in + inStepByN * blockIdx.y + inStepByD * blockIdx.x,
           * cout = out + inStepByN * blockIdx.y + inStepByD * blockIdx.x;
    
    bnOffset(&prm, inStepByD * blockIdx.x);
    
    unsigned int i = threadIdx.x;
    while (i < inStepByD){
                      
        cout[i] = (cin[i] - prm.mean[i]) * prm.scale[i] / prm.varce[i] + prm.schift[i];

        i += blockDim.x;
    }
}

__global__ void layerBatchNormInf(snSize insz, snFloat* in, snFloat* out, batchNorm prm){
     
    size_t inStepByD = insz.w * insz.h,     // step out by input
           inStepByN = inStepByD * insz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    snFloat* cin = in + inStepByN * blockIdx.y + inStepByD * blockIdx.x,
           * cout = out + inStepByN * blockIdx.y + inStepByD * blockIdx.x;
    
    bnOffset(&prm, inStepByD * blockIdx.x);

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

        prm.mean[i] = 0;

        snFloat srq = 0.F;
        for (size_t j = 0; j < bsz; ++j){
           
            snFloat cin = in[i + j * sz];

            prm.mean[i] += cin;

            srq += cin * cin;
        }
               
        prm.mean[i] /= bsz;
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
    // blockDim.x <= inSz

    gradIn += insz.w * insz.h * blockIdx.x;
    prm.norm += insz.w * insz.h * blockIdx.x;
    prm.dSchift += insz.w * insz.h * blockIdx.x;
    prm.dScale += insz.w * insz.h * blockIdx.x;

    unsigned int i = threadIdx.x;
    while (i < (insz.w * insz.h)){

        prm.dSchift[i] = 0;

        snFloat dScale = 0.F;
        for (size_t j = 0; j < bsz; ++j){
            
            snFloat cin = gradIn[i + j * sz],
                    norm = prm.norm[i + j * sz];
          
            prm.dSchift[i] += cin;
            
            dScale += cin * norm;
        }
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


void batchNormForward(SN_Base::snSize insz, snFloat* in, snFloat* out, const batchNorm& prm){
        
    calcMeanAndVarce << < insz.d, 256 >> > (insz, in, prm);

    dim3 dimBlock(128);
    dim3 dimGrid(int(insz.d), int(insz.n));

    calcNormAndOut << < dimGrid, dimBlock >> > (insz, in, out, prm);
}

void batchNormBackward(SN_Base::snSize insz, snFloat* gradIn, snFloat* gradOut, const batchNorm& prm){
    // https://kevinzakka.github.io/2016/09/14/batch_normalization/
    

    calcDSchiftAndDScale << <insz.d, 256 >> > (insz, gradIn, prm);
   
    dim3 dimBlock(128);
    dim3 dimGrid(int(insz.d), int(insz.n));

    calcGrOut << < dimGrid, dimBlock >> > (insz, gradIn, gradOut, prm);

    calcSchiftAndScale << <insz.d, 256 >> > (insz, prm);
}


void channelBatchNorm(bool fwBw, bool isLern, const snSize& insz, snFloat* in, snFloat* out, batchNorm prm){
        
    if (!isLern){

        dim3 dimBlock(128);
        dim3 dimGrid(int(insz.d), int(insz.n));

        channelBatchNormInf << <dimGrid, dimBlock >> > (insz, in, out, prm);
    }
    else{

        size_t stepD = insz.w * insz.h,
               stepN = stepD * insz.d,
               bsz = insz.n;

        snFloat* share = nullptr;
        cuAssert(cudaMalloc(&share, stepD * bsz * sizeof(snFloat)));
       
        snSize sz(insz.w, insz.h, 1, insz.n);

        for (size_t i = 0; i < insz.d; ++i){

            snFloat* pSh = share;
            snFloat* pIn = in + stepD * i;
            for (size_t j = 0; j < bsz; ++j){

                cuMemCpyGPU2GPU(stepD, pSh, pIn, true);
                pSh += stepD;
                pIn += stepN;
            }

            if (fwBw)
                batchNormForward(sz, share, share, prm);
            else
                batchNormBackward(sz, share, share, prm);

            pSh = share;
            snFloat* pOut = out + stepD * i;
            for (size_t j = 0; j < bsz; ++j){
                cuMemCpyGPU2GPU(stepD, pOut, pSh, true));
                pSh += stepD;
                pOut += stepN;
            }

            prm.offset(stepD);
            prm.norm += stepD * bsz;
        }
        cuAssert(cudaFree(share));
    }
}

void layerBatchNorm(bool fwBw, bool isLern, const snSize& insz, snFloat* in, snFloat* out, const batchNorm& prm){

    if (!isLern){

        dim3 dimBlock(128);
        dim3 dimGrid(int(insz.d), int(insz.n));

        layerBatchNormInf << <dimGrid, dimBlock >> > (insz, in, out, prm);
    }
    else{ // isLerning
        if (fwBw)
            batchNormForward(insz, in, out, prm);
        else
            batchNormBackward(insz, in, out, prm);
    }
}

