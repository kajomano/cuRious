#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

#include "threads.h"
#include "debug.h"

extern "C"
SEXP cuR_copy_blob( SEXP src_r, SEXP dest_r, SEXP l_r ){
  double* src  = REAL( src_r );
  double* dest = REAL( dest_r );
  int l        = Rf_asInteger( l_r );

  memcpy( dest, src, l*sizeof(double) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// Various memory allocations/deallocations
// Tensor - device memory
// Stage  - pinned host memory
// Buffer - non-pinned host memory

void cuR_finalize_tensor( SEXP tens_r ){
  float* tens_dev = (float*)R_ExternalPtrAddr( tens_r );

  if( tens_dev ){
#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Finalizing tensor\n", (void*) tens_dev );
#endif

    // Free memory
    cudaFree( tens_dev );
    R_ClearExternalPtr( tens_r );
  }
}

extern "C"
SEXP cuR_create_tensor( SEXP l_r ){
  float* tens_dev;
  cudaError_t cuda_stat;
  int l = Rf_asInteger( l_r );

  cudaTry( cudaMalloc( (void**)&tens_dev, l*sizeof(float) ) )

#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Creating tensor\n", (void*)tens_dev );
#endif

  // Wrap pointer in a SEXP and register a finalizer
  SEXP tens_r = Rf_protect( R_MakeExternalPtr( tens_dev, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( tens_r, cuR_finalize_tensor, TRUE );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_destroy_tensor( SEXP tens_r ){
  cuR_finalize_tensor( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// ----------------------------------------------------------

void cuR_finalize_stage( SEXP stage_r ){
  float* stage = (float*)R_ExternalPtrAddr( stage_r );

  // Destroy context and free memory!
  // Clear R object too
  if( stage ){
#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Finalizing stage\n", (void*)stage );
#endif

    cudaFreeHost( stage );
    R_ClearExternalPtr( stage_r );
  }
}

extern "C"
SEXP cuR_create_stage( SEXP l_r ){
  float* stage;
  cudaError_t cuda_stat;
  int l = Rf_asInteger( l_r );

  cudaTry( cudaHostAlloc( (void**)&stage, l*sizeof(float), cudaHostAllocDefault) )

#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Creating stage\n", (void*) stage );
#endif

  SEXP ret_r = Rf_protect( R_MakeExternalPtr( stage, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ret_r, cuR_finalize_stage, TRUE );

  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_stage( SEXP stage_r ){
  cuR_finalize_stage( stage_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// ----------------------------------------------------------

void cuR_finalize_buffer( SEXP buffer_r ){
  float* buffer = (float*)R_ExternalPtrAddr( buffer_r );

  if( buffer ){
#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Finalizing buffer\n", (void*)buffer );
#endif

    delete[] buffer;
    R_ClearExternalPtr( buffer_r );
  }
}

extern "C"
SEXP cuR_create_buffer( SEXP l_r ){
  int l = Rf_asInteger( l_r );

  float* buffer = new float[l];

#ifdef DEBUG_PRINTS
  Rprintf( "<%p> Creating buffer\n", (void*)buffer );
#endif

  SEXP ret_r = Rf_protect( R_MakeExternalPtr( buffer, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ret_r, cuR_finalize_buffer, TRUE );

  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_buffer( SEXP buffer_r ){
  cuR_finalize_buffer( buffer_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// Pull-push ================================================

extern "C"
SEXP cuR_push_preproc( SEXP data_r, SEXP l_r, SEXP buff_r, SEXP threads_r ){
  // Recover pointers and length
  double* data = REAL( data_r );
  int l        = Rf_asInteger( l_r );
  int threads  = Rf_asInteger( threads_r );
  float* buff  = (float*)R_ExternalPtrAddr( buff_r );

  // Convert values to float
  cuR_conv_2_float( data, buff, l, threads );

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_push_fetch( SEXP buff_r, SEXP l_r, SEXP tens_r ){
  float* buff  = (float*)R_ExternalPtrAddr( buff_r );
  int l        = Rf_asInteger( l_r );
  float* tens  = (float*)R_ExternalPtrAddr( tens_r );
  cudaError_t cuda_stat;

#ifdef DEBUG_PRINTS
  Rprintf( "<%p> Pushing tensor\n", (void*) tens );
#endif

  cudaTry( cudaMemcpy( tens, buff, l*sizeof(float), cudaMemcpyHostToDevice ) )

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_push_fetch_async( SEXP stage_r, SEXP l_r, SEXP tens_r, SEXP stream_r ){
  float* stage         = (float*)R_ExternalPtrAddr( stage_r );
  int l                = Rf_asInteger( l_r );
  float* tens          = (float*)R_ExternalPtrAddr( tens_r );
  cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
  cudaError_t cuda_stat;

#ifdef DEBUG_PRINTS
  Rprintf( "<%p> Async pushing tensor\n", (void*) tens );
#endif

  cudaTry( cudaMemcpyAsync( tens, stage, l*sizeof(float), cudaMemcpyHostToDevice, *stream ) )

  // Flush for WDDM
  cudaStreamQuery(0);

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// ----------------------------------------------------------

extern "C"
SEXP cuR_pull_prefetch( SEXP buff_r, SEXP l_r, SEXP tens_r ){
  float* buff  = (float*)R_ExternalPtrAddr( buff_r );
  int l        = Rf_asInteger( l_r );
  float* tens  = (float*)R_ExternalPtrAddr( tens_r );
  cudaError_t cuda_stat;

#ifdef DEBUG_PRINTS
  Rprintf( "<%p> Pulling tensor\n", (void*)tens );
#endif

  cudaTry( cudaMemcpy( buff, tens, l*sizeof(float), cudaMemcpyDeviceToHost ) )

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_pull_prefetch_async( SEXP stage_r, SEXP l_r, SEXP tens_r, SEXP stream_r ){
  // Recover pointers and length
  float* stage         = (float*)R_ExternalPtrAddr( stage_r );
  int l                = Rf_asInteger( l_r );
  float* tens          = (float*)R_ExternalPtrAddr( tens_r );
  cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
  cudaError_t cuda_stat;

#ifdef DEBUG_PRINTS
  Rprintf( "<%p> Async pulling tensor\n", (void*)tens );
#endif

  cudaTry( cudaMemcpyAsync( stage, tens, l*sizeof(float), cudaMemcpyDeviceToHost, *stream ) );

  // Flush for WDDM
  cudaStreamQuery(0);

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_pull_proc( SEXP data_r, SEXP l_r, SEXP buff_r, SEXP threads_r ){
  // Recover pointers and length
  double* data = REAL( data_r );
  int l        = Rf_asInteger( l_r );
  int threads  = Rf_asInteger( threads_r );
  float* buff  = (float*)R_ExternalPtrAddr( buff_r );

  // Convert values to double
  cuR_conv_2_double( buff, data, l, threads );

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}
