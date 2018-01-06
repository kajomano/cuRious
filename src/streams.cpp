#include <cuda.h>
#include <cuda_runtime_api.h>

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

#include "debug.h"

extern "C"
SEXP cuR_sync_device(){
  cudaError_t cuda_stat;
  cudaTry( cudaDeviceSynchronize() )

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_sync_cuda_stream( SEXP stream_r ){
  cudaError_t cuda_stat;
  cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
  cudaTry( cudaStreamSynchronize( *stream ) )

    // Return something that is not null
    SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

void cuR_finalize_cuda_stream( SEXP ptr ){
  cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( ptr );

  // Destroy context and free memory!
  // Clear R object too
  if( stream ){
#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Finalizing stream\n", (void*)stream );
#endif

    cudaStreamDestroy( *stream );
    delete[] stream;
    R_ClearExternalPtr( ptr );
  }
}

extern "C"
SEXP cuR_deactivate_cuda_stream( SEXP ptr ){
  cuR_finalize_cuda_stream( ptr );
  return R_NilValue;
}

extern "C"
SEXP cuR_activate_cuda_stream(){
  cudaStream_t* stream = new cudaStream_t;
#ifdef DEBUG_PRINTS
  Rprintf( "<%p> Creating stream\n", (void*)stream );
#endif

  cudaError_t cuda_stat;
  cudaTry( cudaStreamCreate( stream ) )

  // Return to R with an external pointer SEXP
  SEXP ptr = Rf_protect( R_MakeExternalPtr( stream, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ptr, cuR_finalize_cuda_stream, TRUE );

  Rf_unprotect(1);
  return ptr;
}
