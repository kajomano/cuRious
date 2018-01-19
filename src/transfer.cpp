#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>

#define R_NO_REMAP 1

#include <R.h>
#include <Rinternals.h>

#include "threads.h"
#include "debug.h"

extern "C"
SEXP cuR_transf_0_0( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP rsrc_r, SEXP rdst_r, SEXP threads_r ){

  double* src  = REAL( src_r );
  double* dst  = REAL( dst_r );
  int* dims    = INTEGER( dims_r );
  int* rsrc    = ( R_NilValue == rsrc_r ) ? NULL : INTEGER( rsrc_r );
  int* rdst    = ( R_NilValue == rdst_r ) ? NULL : INTEGER( rdst_r );
  int threads  = Rf_asInteger( threads_r );

  cuR_threaded_dd( src, dst, dims, rsrc, rdst, threads );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// ==============================================================================

extern "C"
SEXP cuR_copy_obj( SEXP src_r, SEXP dest_r, SEXP l_r ){
  double* src  = REAL( src_r );
  double* dest = REAL( dest_r );
  int l        = Rf_asInteger( l_r );

  memcpy( dest, src, l*sizeof(double) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

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
