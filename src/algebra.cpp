#include "common_cpp.h"
#include "algebra_cu.h"

extern "C"
SEXP ewop( SEXP ptr_l, SEXP ptr_r, SEXP ptr_res, SEXP op_r, SEXP n_dims_r, SEXP dims_r ) {
  // Convert all SEXPs to proper pointers to GPU data
  float* dev_l   = (float*)R_ExternalPtrAddr(ptr_l);
  float* dev_r   = (float*)R_ExternalPtrAddr(ptr_r);
  float* dev_res = (float*)R_ExternalPtrAddr(ptr_res);

  // Convert operator
  int op = Rf_asInteger(op_r);

  return R_NilValue;

  // // Calculate vector length
  // int l = get_tensor_length( Rf_asInteger(n_dims_r), INTEGER(dims_r) );
  //
  // // Call addition kernel
  // elem_wise_add_host( dev_l, dev_r, L, dev_res );
  //
  // return R_NilValue;
}

