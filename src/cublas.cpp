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
#ifdef DEBUG_PRINTS
    Rprintf( "Finalizing handle at <%p>\n", (void*)handle );
#endif

    cublasDestroy( *handle );
    delete[] handle;
    R_ClearExternalPtr( ptr );
  }
}

extern "C"
SEXP cuR_create_cublas_handle(){
  cublasHandle_t* handle = new cublasHandle_t;
#ifdef DEBUG_PRINTS
  Rprintf( "Creating handle at <%p>\n", (void*)handle );
#endif

  // cublasTry is a macro defined in debug.h, you need to create the variable
  // 'stat' beforehand
  cublasStatus_t stat;
  cublasTry( cublasCreate( handle ) )

  // Return to R with an external pointer SEXP
  SEXP ptr = Rf_protect( R_MakeExternalPtr( handle, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ptr, cuR_finalize_cublas_handle, TRUE );

  Rf_unprotect(1);
  return ptr;
}

extern "C"
SEXP cuR_cublas_saxpy( SEXP tens_x_r, SEXP tens_y_r, SEXP l_r, SEXP al_r, SEXP handle_r ){
  // Recover handle
  cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( handle_r );

  // Recover tensors, the length and the scalar
  int l = Rf_asInteger( l_r );
  float* tens_x = (float*)R_ExternalPtrAddr( tens_x_r );
  float* tens_y = (float*)R_ExternalPtrAddr( tens_y_r );
  float al = (float)Rf_asReal( al_r );

  // Do the operation, the results go into tens_y
  cublasStatus_t stat;
  cublasTry( cublasSaxpy( *handle, l, &al, tens_x, 1, tens_y, 1 ) )

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}
