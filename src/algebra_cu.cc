#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

#include "common.h"

__global__
void elem_wise_add_global( float* dev_l, float* dev_r, int l, float* dev_res )
{
  for(int i = 0; i < l; i++){
    dev_res[i] = dev_l[i] + dev_r[i];
  }
}

extern "C"
void elem_wise_add_host(  float* dev_l, float* dev_r, int l, float* dev_res ){
  elem_wise_add_global<<<1, 1>>>(dev_l, dev_r, l, dev_res);
  cudaDeviceSynchronize();

}
