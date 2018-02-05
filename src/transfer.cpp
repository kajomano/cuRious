#include "common.h"
#include <cstring>

template <typename s, typename d>
void cuR_transfer_host_host( s* src, d* dst, int* dims, int osrc, int odst, int* csrc, int* cdst ){
  // Offsets disable column subsetting for safety
  if( osrc ){
    src  = src + (osrc * dims[0]);
    csrc = NULL;
  }

  if( odst ){
    dst  = dst + (odst * dims[0]);
    cdst = NULL;
  }

  // Copy
  if( !csrc && !cdst ){
    // No subsetting
    int l = dims[0]*dims[1];
    for( int j = 0; j < l; j++ ){
      dst[j] = (d)src[j];
    }
  }else if( csrc && cdst ){
    // Both subsetted
    int dst_off, src_off;
    for( int i = 0; i < dims[1]; i ++ ){
      dst_off = (cdst[i]-1)*dims[0];
      src_off = (csrc[i]-1)*dims[0];
      for( int j = 0; j < dims[0]; j++ ){
        dst[dst_off+j] = (d)src[src_off+j];
      }
    }
  }else if( !csrc ){
    // Destination subsetted
    int dst_off;
    int src_off = 0;
    for( int i = 0; i < dims[1]; i ++ ){
      dst_off = (cdst[i]-1)*dims[0];
      for( int j = 0; j < dims[0]; j++ ){
        dst[dst_off+j] = (d)src[src_off+j];
      }
      src_off += dims[0];
    }
  }else{
    // Source subsetted
    int dst_off = 0;
    int src_off;
    for( int i = 0; i < dims[1]; i ++ ){
      src_off = (csrc[i]-1)*dims[0];
      for( int j = 0; j < dims[0]; j++ ){
        dst[dst_off+j] = (d)src[src_off+j];
      }
      dst_off += dims[0];
    }
  }
}

// -----------------------------------------------------------------------------

extern "C"
SEXP cuR_transfer_0_0_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  double* src = REAL( src_r );
  double* dst = REAL( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<double, double>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_0_0_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  int* src    = INTEGER( src_r );
  int* dst    = INTEGER( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<int, int>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_0_0_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  int* src    = LOGICAL( src_r );
  int* dst    = LOGICAL( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<int, int>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_0_12_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  double* src = REAL( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<double, float>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_0_12_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  int* src    = INTEGER( src_r );
  int* dst    = (int*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<int, int>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_0_12_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  int* src    = LOGICAL( src_r );
  bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<int, bool>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_12_0_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  double* dst = REAL( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<float, double>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_12_0_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  int* src    = (int*)R_ExternalPtrAddr( src_r );
  int* dst    = INTEGER( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<int, int>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_12_0_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  bool* src   = (bool*)R_ExternalPtrAddr( src_r );
  int* dst    = LOGICAL( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<bool, int>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_12_12_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<float, float>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_12_12_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  int* src    = (int*)R_ExternalPtrAddr( src_r );
  int* dst    = (int*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<int, int>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_12_12_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  bool* src   = (bool*)R_ExternalPtrAddr( src_r );
  bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer_host_host<bool, bool>( src, dst, dims, osrc, odst, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

#ifndef CUDA_EXCLUDE

extern "C"
SEXP cuR_transfer_1_3_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyHostToDevice ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_1_3_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){

  int* src    = (int*)R_ExternalPtrAddr( src_r );
  int* dst    = (int*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  cudaTry( cudaMemcpy( dst, src, l*sizeof(int), cudaMemcpyHostToDevice ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_1_3_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){

  bool* src   = (bool*)R_ExternalPtrAddr( src_r );
  bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  cudaTry( cudaMemcpy( dst, src, l*sizeof(bool), cudaMemcpyHostToDevice ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_3_1_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyDeviceToHost ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_3_1_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){

  int* src    = (int*)R_ExternalPtrAddr( src_r );
  int* dst    = (int*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  cudaTry( cudaMemcpy( dst, src, l*sizeof(int), cudaMemcpyDeviceToHost ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_3_1_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){

  bool* src   = (bool*)R_ExternalPtrAddr( src_r );
  bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  cudaTry( cudaMemcpy( dst, src, l*sizeof(bool), cudaMemcpyDeviceToHost ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_2_3_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

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
SEXP cuR_transfer_2_3_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){

  int* src    = (int*)R_ExternalPtrAddr( src_r );
  int* dst    = (int*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  if( stream_r != R_NilValue ){
    cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
    cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(int), cudaMemcpyHostToDevice, *stream ) );

    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaTry( cudaMemcpy( dst, src, l*sizeof(int), cudaMemcpyHostToDevice ) );
  }

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_2_3_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){

  bool* src   = (bool*)R_ExternalPtrAddr( src_r );
  bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  if( stream_r != R_NilValue ){
    cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
    cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(bool), cudaMemcpyHostToDevice, *stream ) );

    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaTry( cudaMemcpy( dst, src, l*sizeof(bool), cudaMemcpyHostToDevice ) );
  }

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_3_2_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

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
SEXP cuR_transfer_3_2_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){

  int* src    = (int*)R_ExternalPtrAddr( src_r );
  int* dst    = (int*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  if( stream_r != R_NilValue ){
    cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
    cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(int), cudaMemcpyDeviceToHost, *stream ) );

    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaTry( cudaMemcpy( dst, src, l*sizeof(int), cudaMemcpyDeviceToHost ) );
  }

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_3_2_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){

  bool* src   = (bool*)R_ExternalPtrAddr( src_r );
  bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  if( stream_r != R_NilValue ){
    cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
    cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(bool), cudaMemcpyDeviceToHost, *stream ) );

    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaTry( cudaMemcpy( dst, src, l*sizeof(bool), cudaMemcpyDeviceToHost ) );
  }

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_3_3_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){

  float* src  = (float*)R_ExternalPtrAddr( src_r );
  float* dst  = (float*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

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

extern "C"
SEXP cuR_transfer_3_3_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){

  int* src    = (int*)R_ExternalPtrAddr( src_r );
  int* dst    = (int*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  if( stream_r != R_NilValue ){
    cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
    cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(int), cudaMemcpyDeviceToDevice, *stream ) );

    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaTry( cudaMemcpy( dst, src, l*sizeof(int), cudaMemcpyDeviceToDevice ) );
    cudaTry( cudaDeviceSynchronize() )
  }

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_transfer_3_3_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){

  bool* src   = (bool*)R_ExternalPtrAddr( src_r );
  bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
  int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
  int* dims   = INTEGER( dims_r );
  int l       = dims[0]*dims[1];

  src  = src + (osrc * dims[0]);
  dst  = dst + (odst * dims[0]);

  if( stream_r != R_NilValue ){
    cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
    cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(bool), cudaMemcpyDeviceToDevice, *stream ) );

    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaTry( cudaMemcpy( dst, src, l*sizeof(bool), cudaMemcpyDeviceToDevice ) );
    cudaTry( cudaDeviceSynchronize() )
  }

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

#endif
