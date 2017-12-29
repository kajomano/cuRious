#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

#include "debug.h"

// This is a good example how to access the handle in other functions
// Also a good example how to finalize things, this function can be called
// from other functions without fear double-finalization
void cuR_finalize_cublas_handle( SEXP ptr ){
  cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( ptr );

  // Destroy context and free memory!
  // Clear R object too
  if( handle ){
    cublasDestroy( *handle );
    delete[] handle;
    R_ClearExternalPtr( ptr );

#ifdef DEBUG_PRINTS
    Rprintf( "Finalized cublas handle\n" );
#endif
  }
}

extern "C"
SEXP cuR_create_cublas_handle(){
  cublasHandle_t* handle = new cublasHandle_t;
  cublasStatus_t stat = cublasCreate( handle );

  // Trying to do error handling
  if( stat != CUBLAS_STATUS_SUCCESS ){
    return R_NilValue;
  }

  // Return to R with an external pointer SEXP
  SEXP ptr = Rf_protect( R_MakeExternalPtr( handle, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ptr, cuR_finalize_cublas_handle, TRUE );

  Rf_unprotect(1);
  return ptr;
}
