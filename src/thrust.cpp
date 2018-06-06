#include "thrust.h"

extern "C"
SEXP cuR_thrust_pow2( SEXP A_ptr_r,
                      SEXP B_ptr_r,
                      SEXP dims_r,
                      SEXP A_span_off_r,   // Optional
                      SEXP B_span_off_r,   // Optional
                      SEXP stream_ptr_r ){ // Optional

  float* A_ptr    = (float*)R_ExternalPtrAddr( A_ptr_r );
  float* B_ptr    = (float*)R_ExternalPtrAddr( B_ptr_r );
  int*   dims     = INTEGER( dims_r );

  int A_span_off  = ( R_NilValue == A_span_off_r ) ? 0:
    ( Rf_asInteger( A_span_off_r ) - 1 );

  int B_span_off  = ( R_NilValue == B_span_off_r ) ? 0:
    ( Rf_asInteger( B_span_off_r ) - 1 );

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*) R_ExternalPtrAddr( stream_ptr_r );

  // Offsets
  A_ptr = A_ptr + A_span_off * dims[0];
  B_ptr = B_ptr + B_span_off * dims[0];

  cuR_thrust_pow2_cu( A_ptr, B_ptr, dims, stream_ptr );

  if( stream_ptr ){
    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaTry( cudaDeviceSynchronize() );
  }

  return R_NilValue;
}

// extern "C"
// SEXP cuB_thrust_cmins( SEXP prod_r, SEXP dims_r, SEXP quant_r ) {
//   float* prod  = (float*)R_ExternalPtrAddr( prod_r );
//   int*   dims  = INTEGER( dims_r );
//   int*   quant = (int*)R_ExternalPtrAddr( quant_r );
//
//   cuB_thrust_cmins_cu( prod, dims, quant );
//
//   // Return something that is not null
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuB_thrust_table( SEXP quant_r, SEXP perm_r, SEXP temp_quant_r, SEXP dims_r, SEXP weights_r, SEXP dims_weights_r ) {
//   int* quant        = (int*)R_ExternalPtrAddr( quant_r );
//   int* perm         = (int*)R_ExternalPtrAddr( perm_r );
//   int* temp_quant   = (int*)R_ExternalPtrAddr( temp_quant_r );
//   int* dims         = INTEGER( dims_r );
//
//   int* weights      = (int*)R_ExternalPtrAddr( weights_r );
//   int* dims_weights = INTEGER( dims_weights_r );
//
//   cuB_thrust_table_cu( quant, perm, temp_quant, dims, weights, dims_weights );
//
//   // Return something that is not null
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
