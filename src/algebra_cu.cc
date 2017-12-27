#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

#include "common.h"

__global__
void elem_wise_add_global( float* dev_l, float* dev_r, int l, float* dev_res )
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < l; i += stride)
    dev_res[i] = dev_l[i] + dev_r[i];
}

extern "C"
void elem_wise_add_host(  float* dev_l, float* dev_r, int l, float* dev_res ){

  int blockSize = 256;
  int numBlocks = (l + blockSize - 1) / blockSize;

  elem_wise_add_global<<<numBlocks, blockSize>>>(dev_l, dev_r, l, dev_res);

  cudaDeviceSynchronize();
}
