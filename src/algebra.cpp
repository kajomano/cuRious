#include "algebra.h"

#ifndef CUDA_EXCLUDE

cudaStream_t* cuR_alg_recover_stream( SEXP stream_r ){
  if( stream_r != R_NilValue ){
    debugPrint( Rprintf( "Async alg call\n" ) );
    return (cudaStream_t*)R_ExternalPtrAddr( stream_r );
  }else{
    debugPrint( Rprintf( "Sync alg call\n" ) );
    return NULL;
  }
}

extern "C"
SEXP cuR_alg_saxpy( SEXP tens_x_r, SEXP tens_y_r, SEXP l_r, SEXP al_r, SEXP stream_r ) {
  // Recover tensors, the length and the scalar
  int l = Rf_asInteger( l_r );
  float* tens_x = (float*)R_ExternalPtrAddr( tens_x_r );
  float* tens_y = (float*)R_ExternalPtrAddr( tens_y_r );
  float al = (float)Rf_asReal( al_r );

  // If a stream is given, set it before the calculation
  cudaStream_t* stream = cuR_alg_recover_stream( stream_r );

  // Call the op
  cuR_alg_saxpy_cu(  tens_x, tens_y, l, al, stream );

  // Flush for WDDM
  cudaTry( cudaStreamQuery(0) );

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

#endif
