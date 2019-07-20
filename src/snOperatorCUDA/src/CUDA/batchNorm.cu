
#include <cuda_runtime.h>

#include "../stdafx.h"

using namespace SN_Base;

__device__ void bnOffset(batchNorm& bn, size_t offs){
    bn.mean += offs;
    bn.varce += offs;
    bn.scale += offs;
    bn.dScale += offs;
    bn.schift += offs;
    bn.dSchift += offs;
}

__global__ void channelBatchNormInf(const snSize& insz, snFloat* in, snFloat* out, batchNorm prm){
    
    size_t inStepByD = insz.w * insz.h,     // step out by input
           inStepByN = inStepByD * insz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size
        
    snFloat* cin = in + inStepByN * blockIdx.y + inStepByD * blockIdx.x,
           * cout = out + inStepByN * blockIdx.y + inStepByD * blockIdx.x;
    
    bnOffset(prm, inStepByD * blockIdx.x);
    
    unsigned int i = threadIdx.x;
    while (i < inStepByD){
                      
        cout[i] = (cin[i] - prm.mean[i]) * prm.scale[i] / prm.varce[i] + prm.schift[i];

        i += blockDim.x;
    }
}

__global__ void layerBatchNormInf(const snSize& insz, snFloat* in, snFloat* out, batchNorm prm){
     
    size_t inStepByD = insz.w * insz.h,     // step out by input
           inStepByN = inStepByD * insz.d;  // step out by batch       

    // gridDim.x - number of out layers
    // gridDim.y - batch size

    snFloat* cin = in + inStepByN * blockIdx.y + inStepByD * blockIdx.x,
           * cout = out + inStepByN * blockIdx.y + inStepByD * blockIdx.x;
    
    unsigned int i = threadIdx.x;
    while (i < inStepByD){

        cout[i] = (cin[i] - prm.mean[i]) * prm.scale[i] / prm.varce[i] + prm.schift[i];

        i += blockDim.x;
    }
}

//
//void batchNormForward(const SN_Base::snSize& insz, snFloat* in, snFloat* out, batchNorm prm){
//
//    size_t inSz = insz.w * insz.h * insz.d, bsz = insz.n;
//
//    /// μ = 1/n * ∑x
//    cblas_sgemv(CBLAS_ORDER::CblasRowMajor,
//        CBLAS_TRANSPOSE::CblasTrans,
//        blasint(bsz),
//        blasint(inSz),
//        1.F / bsz,
//        in,
//        blasint(inSz),
//        prm.onc,
//        blasint(1),
//        0.F,
//        prm.mean,
//        blasint(1));
//
//    /// varce = sqrt(∑xx - mean^2 + e)
//    for (size_t i = 0; i < inSz; ++i){
//
//        snFloat* cin = in + i, srq = 0.F;
//        for (size_t j = 0; j < bsz; ++j){
//            srq += cin[0] * cin[0];
//            cin += inSz;
//        }
//        prm.varce[i] = sqrt(srq / bsz - prm.mean[i] * prm.mean[i] + 0.00001F);
//    }
//
//    /// norm = (in - mean) / varce
//    /// y = ^x * γ + β
//    for (size_t j = 0; j < bsz; ++j){
//
//        snFloat* cin = in + j * inSz, *cout = out + j * inSz, *norm = prm.norm + j * inSz;
//
//        for (size_t i = 0; i < inSz; ++i){
//            norm[i] = (cin[i] - prm.mean[i]) / prm.varce[i];
//            cout[i] = norm[i] * prm.scale[i] + prm.schift[i];
//        }
//    }
//
//}
//
//void batchNormBackward(const SN_Base::snSize& insz, snFloat* gradIn, snFloat* gradOut, batchNorm prm){
//    // https://kevinzakka.github.io/2016/09/14/batch_normalization/
//
//    size_t inSz = insz.w * insz.h * insz.d, bsz = insz.n;
//
//    /// ∂f/∂β = ∑∂f/∂y
//    cblas_sgemv(CBLAS_ORDER::CblasRowMajor,
//        CBLAS_TRANSPOSE::CblasTrans,
//        blasint(bsz),
//        blasint(inSz),
//        1.F,
//        gradIn,
//        blasint(inSz),
//        prm.onc,
//        blasint(1),
//        0.F,
//        prm.dSchift,
//        blasint(1));
//
//    /// ∂f/∂γ = ∑∂f/∂y ⋅ ^x
//    for (size_t i = 0; i < inSz; ++i){
//
//        snFloat* igr = gradIn + i, *norm = prm.norm + i, dScale = 0.F;
//        for (size_t j = 0; j < bsz; ++j){
//
//            dScale += igr[0] * norm[0];
//
//            norm += inSz;
//            igr += inSz;
//        }
//        prm.dScale[i] = dScale;
//    }
//
//    /// ∂f/∂x = (m⋅γ⋅∂f/∂y − γ⋅∂f/∂β − ^x⋅γ⋅∂f/∂γ) / m⋅σ2
//    for (size_t j = 0; j < bsz; ++j){
//
//        snFloat* igr = gradIn + j * inSz, *ogr = gradOut + j * inSz, *norm = prm.norm + j * inSz;
//        for (size_t i = 0; i < inSz; ++i)
//            ogr[i] = prm.scale[i] * (igr[i] * bsz - prm.dSchift[i] - norm[i] * prm.dScale[i]) / (prm.varce[i] * bsz);
//    }
//    for (size_t i = 0; i < inSz; ++i){
//        prm.schift[i] -= prm.dSchift[i] * prm.lr;
//        prm.scale[i] -= prm.dScale[i] * prm.lr;
//    }
//
//}


void channelBatchNorm(bool fwBw, bool isLern, const snSize& insz, snFloat* in, snFloat* out, batchNorm prm){
        
    if (!isLern){

        dim3 dimBlock(128);
        dim3 dimGrid(int(insz.d), int(insz.n));

        channelBatchNormInf << <dimGrid, dimBlock >> > (insz, in, out, prm);
    }
   /* else{

        size_t stepD = insz.w * insz.h,
               stepN = stepD * insz.d,
               bsz = insz.n;

        snFloat* share = (snFloat*)calloc(stepD * bsz, sizeof(snFloat));
        snSize sz(insz.w, insz.h, 1, insz.n);

        for (size_t i = 0; i < insz.d; ++i){

            snFloat* pSh = share;
            snFloat* pIn = in + stepD * i;
            for (size_t j = 0; j < bsz; ++j){

                memcpy(pSh, pIn, stepD * sizeof(snFloat));
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
                memcpy(pOut, pSh, stepD * sizeof(snFloat));
                pSh += stepD;
                pOut += stepN;
            }

            prm.offset(stepD);
            prm.norm += stepD * bsz;
        }
        free(share);
    }*/
}

void layerBatchNorm(bool fwBw, bool isLern, const snSize& insz, snFloat* in, snFloat* out, const batchNorm& prm){

    if (!isLern){

        dim3 dimBlock(128);
        dim3 dimGrid(int(insz.d), int(insz.n));

        layerBatchNormInf << <dimGrid, dimBlock >> > (insz, in, out, prm);
    }
    //else{ // isLerning
    //    if (fwBw)
    //        batchNormForward(insz, in, out, prm);
    //    else
    //        batchNormBackward(insz, in, out, prm);
    //}
}

