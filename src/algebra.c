#include <cuda.h>
#include <cuda_runtime_api.h>

#include "common.h"

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void dev_add( float* dev_l, float* dev_r, int l, float* dev_res )
{
  for(int i = 0; i < l; i++){
    dev_res[i] = dev_l[i] + dev_r[i];
  }
}

SEXP elem_wise_add( SEXP ptr_l, SEXP ptr_r, SEXP l, SEXP ptr_res ) {
  // Convert all SEXPs to proper pointers to GPU data
  float* dev_l   = (float*)R_ExternalPtrAddr(ptr_l);
  float* dev_r   = (float*)R_ExternalPtrAddr(ptr_r);
  float* dev_res = (float*)R_ExternalPtrAddr(ptr_res);

  // Save vector length
  int L = Rf_asInteger(l);

  dev_add<<<1, 1>>>(dev_l, dev_r, L, dev_res);

  return R_NilValue;
}

