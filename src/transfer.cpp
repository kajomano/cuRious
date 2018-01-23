#include "common.h"
#include "threads.h"
#include <cstring>
#include <omp.h>

void cuR_dd( double* src, double* dst, int* dims, int* csrc, int* cdst ){
  int l = dims[0]*dims[1];

  if( !csrc && !cdst ){
    // No subsetting
    memcpy( dst,
            src,
            l*sizeof(double) );

  }else if( csrc && cdst ){
    // Both subsetted
    for( int i = 0; i < dims[1]; i ++ ){
      memcpy( dst+(cdst[i]-1)*dims[0],
              src+(csrc[i]-1)*dims[0],
              dims[0]*sizeof(double) );
    }
  }else if( !csrc ){
    // Destination subsetted
    for( int i = 0; i < dims[1]; i ++ ){
      memcpy( dst+(cdst[i]-1)*dims[0],
              src+i*dims[0],
              dims[0]*sizeof(double) );
    }
  }else{
    // Source subsetted
    for( int i = 0; i < dims[1]; i ++ ){
      memcpy( dst+i*dims[0],
              src+(csrc[i]-1)*dims[0],
              dims[0]*sizeof(double) );
    }
  }
}

void cuR_ff( float* src, float* dst, int* dims, int* csrc, int* cdst ){
  int l = dims[0]*dims[1];

  if( !csrc && !cdst ){
    // No subsetting
    memcpy( dst,
            src,
            l*sizeof(float) );

  }else if( csrc && cdst ){
    // Both subsetted
    for( int i = 0; i < dims[1]; i ++ ){
      memcpy( dst+(cdst[i]-1)*dims[0],
              src+(csrc[i]-1)*dims[0],
              dims[0]*sizeof(float) );
    }
  }else if( !csrc ){
    // Destination subsetted
    for( int i = 0; i < dims[1]; i ++ ){
      memcpy( dst+(cdst[i]-1)*dims[0],
              src+i*dims[0],
              dims[0]*sizeof(float) );
    }
  }else{
    // Source subsetted
    for( int i = 0; i < dims[1]; i ++ ){
      memcpy( dst+i*dims[0],
              src+(csrc[i]-1)*dims[0],
              dims[0]*sizeof(float) );
    }
  }
}

// -----------------------------------------------------------------------------

extern "C"
SEXP cuR_transf_0_0( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r ){

  double* src = REAL( src_r );
  double* dst = REAL( dst_r );
  int* dims   = INTEGER( dims_r );
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_dd( src, dst, dims, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transf_12_12( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_ff( src, dst, dims, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transf_0_12( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP threads_r ){

  double* src = REAL( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );
  int threads  = Rf_asInteger( threads_r );

  // cuR_threaded_df( src, dst, dims, csrc, cdst, threads );
  int l = dims[0]*dims[1];
  omp_set_num_threads(threads);
  #pragma omp parallel
  for( int i = 0; i < l; i++ ){
    dst[i] = (float)src[i];
  }

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transf_12_0( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP threads_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  double* dst = REAL( dst_r );
  int* dims   = INTEGER( dims_r );
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );
  int threads = Rf_asInteger( threads_r );

  cuR_threaded_fd( src, dst, dims, csrc, cdst, threads );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

#ifndef CUDA_EXCLUDE

extern "C"
SEXP cuR_transf_1_3( SEXP src_r, SEXP dst_r, SEXP dims_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyHostToDevice ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transf_3_1( SEXP src_r, SEXP dst_r, SEXP dims_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyDeviceToHost ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transf_2_3( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP stream_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  if( stream_r != R_NilValue ){
    cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
    cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(float), cudaMemcpyHostToDevice, *stream ) );

    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyHostToDevice ) );
  }

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transf_3_2( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP stream_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  if( stream_r != R_NilValue ){
    cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
    cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(float), cudaMemcpyDeviceToHost, *stream ) );

    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyDeviceToHost ) );
  }

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transf_3_3( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP stream_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  if( stream_r != R_NilValue ){
    cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
    cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(float), cudaMemcpyDeviceToDevice, *stream ) );

    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyDeviceToDevice ) );
    cudaTry( cudaDeviceSynchronize() )
  }

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

#endif
