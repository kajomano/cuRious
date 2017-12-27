// #include <cuda.h>
// #include <cuda_runtime_api.h>

#include "common.h"
#include "algebra_cu.h"

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

extern "C"
SEXP elem_wise_add( SEXP ptr_l, SEXP ptr_r, SEXP l, SEXP ptr_res ) {
  // Convert all SEXPs to proper pointers to GPU data
  float* dev_l   = (float*)R_ExternalPtrAddr(ptr_l);
  float* dev_r   = (float*)R_ExternalPtrAddr(ptr_r);
  float* dev_res = (float*)R_ExternalPtrAddr(ptr_res);

  // Save vector length
  int L = Rf_asInteger(l);

  // Call addition kernel
  elem_wise_add_host( dev_l, dev_r, L, dev_res );

  return R_NilValue;
}

