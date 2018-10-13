#!/bin/bash

# $1 - SN_CUDA or SNCUDNN
# $2 - src dir 
# $3 - out bin dir 

if [[ -z "$1" || -z "$2" || -z "$3" ]]; then
  echo "No parameters found"
  echo "1 - SN_CUDA or SNCUDNN"
  echo "2 - src dir"
  echo "3 - out bin dir"
  exit 1
fi

lib='cublas'
if [ $1 = "SN_CUDNN" ]; then 
  lib='cublas,cudnn'
  echo $lib
fi

sdir=$2"/snOperator/src/CUDA"

nvcc $sdir"/convolutionCUDA.cu" $sdir"/deconvolutionCUDA.cu" $sdir"/fullyConnectedCUDA.cu" $sdir"/poolingCUDA.cu" \
--device-c \
--define-macro="$1" \
--include-path="$2" \
--std=c++11 \
--compiler-options -fPIC \
--library="$lib" \
--machine=64 \
--output-directory="$3"
