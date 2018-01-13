#include "algebra_cu.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#define R_NO_REMAP 1

#include <R.h>
#include <Rinternals.h>

#include "debug.h"

cudaStream_t* cuR_alg_recover_stream( SEXP stream_r ){
  if( stream_r != R_NilValue ){
#ifdef DEBUG_PRINTS
    Rprintf( "Async alg call\n" );
#endif

    return (cudaStream_t*)R_ExternalPtrAddr( stream_r );
  }else{
#ifdef DEBUG_PRINTS
    Rprintf( "Sync alg call\n" );
#endif

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
  cudaStreamQuery(0);

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

