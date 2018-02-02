#include "common.h"
#include <cstring>

// void cuR_dd( double* src, double* dst, int* dims, int* csrc, int* cdst ){
//   if( !csrc && !cdst ){
//     // No subsetting
//     memcpy( dst,
//             src,
//             dims[0]*dims[1]*sizeof(double) );
//
//   }else if( csrc && cdst ){
//     // Both subsetted
//     for( int i = 0; i < dims[1]; i ++ ){
//       memcpy( dst+(cdst[i]-1)*dims[0],
//               src+(csrc[i]-1)*dims[0],
//               dims[0]*sizeof(double) );
//     }
//   }else if( !csrc ){
//     // Destination subsetted
//     for( int i = 0; i < dims[1]; i ++ ){
//       memcpy( dst+(cdst[i]-1)*dims[0],
//               src+i*dims[0],
//               dims[0]*sizeof(double) );
//     }
//   }else{
//     // Source subsetted
//     for( int i = 0; i < dims[1]; i ++ ){
//       memcpy( dst+i*dims[0],
//               src+(csrc[i]-1)*dims[0],
//               dims[0]*sizeof(double) );
//     }
//   }
// }

template <typename s, typename d>
void cuR_transfer( s* src, d* dst, int* dims, int osrc, int odst, int* csrc, int* cdst ){
  // Offsets disable column subsetting for safety
  if( osrc ){
    src  = src + osrc;
    csrc = NULL;
  }

  if( odst ){
    dst  = dst + odst;
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
//
// void cuR_fd( float* src, double* dst, int* dims, int* csrc, int* cdst ){
//   if( !csrc && !cdst ){
//     // No subsetting
//     int l = dims[0]*dims[1];
//     for( int j = 0; j < l; j++ ){
//       dst[j] = (double)src[j];
//     }
//   }else if( csrc && cdst ){
//     // Both subsetted
//     int dst_off, src_off;
//     for( int i = 0; i < dims[1]; i ++ ){
//       dst_off = (cdst[i]-1)*dims[0];
//       src_off = (csrc[i]-1)*dims[0];
//       for( int j = 0; j < dims[0]; j++ ){
//         dst[dst_off+j] = (double)src[src_off+j];
//       }
//     }
//   }else if( !csrc ){
//     // Destination subsetted
//     int dst_off;
//     int src_off = 0;
//     for( int i = 0; i < dims[1]; i ++ ){
//       dst_off = (cdst[i]-1)*dims[0];
//       for( int j = 0; j < dims[0]; j++ ){
//         dst[dst_off+j] = (double)src[src_off+j];
//       }
//       src_off += dims[0];
//     }
//   }else{
//     // Source subsetted
//     int dst_off = 0;
//     int src_off;
//     for( int i = 0; i < dims[1]; i ++ ){
//       src_off = (csrc[i]-1)*dims[0];
//       for( int j = 0; j < dims[0]; j++ ){
//         dst[dst_off+j] = (double)src[src_off+j];
//       }
//       dst_off += dims[0];
//     }
//   }
// }
//
// void cuR_ff( float* src, float* dst, int* dims, int* csrc, int* cdst ){
//   if( !csrc && !cdst ){
//     // No subsetting
//     memcpy( dst,
//             src,
//             dims[0]*dims[1]*sizeof(float) );
//
//   }else if( csrc && cdst ){
//     // Both subsetted
//     for( int i = 0; i < dims[1]; i ++ ){
//       memcpy( dst+(cdst[i]-1)*dims[0],
//               src+(csrc[i]-1)*dims[0],
//               dims[0]*sizeof(float) );
//     }
//   }else if( !csrc ){
//     // Destination subsetted
//     for( int i = 0; i < dims[1]; i ++ ){
//       memcpy( dst+(cdst[i]-1)*dims[0],
//               src+i*dims[0],
//               dims[0]*sizeof(float) );
//     }
//   }else{
//     // Source subsetted
//     for( int i = 0; i < dims[1]; i ++ ){
//       memcpy( dst+i*dims[0],
//               src+(csrc[i]-1)*dims[0],
//               dims[0]*sizeof(float) );
//     }
//   }
// }

// -----------------------------------------------------------------------------

extern "C"
SEXP cuR_transfer_0_0_d( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP csrc_r, SEXP cdst_r ){

  double* src = REAL( src_r );
  double* dst = REAL( dst_r );
  int* dims   = INTEGER( dims_r );
  int osrc    = ( R_NilValue == osrc_r ) ? 0 : Rf_asInteger( osrc_r );
  int odst    = ( R_NilValue == odst_r ) ? 0 : Rf_asInteger( odst_r );
  int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
  int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );

  cuR_transfer<double, double>( src, dst, dims, osrc, odst, csrc, cdst );
  // cuR_dd( src, dst, dims, csrc, cdst );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// extern "C"
// SEXP cuR_transf_12_12( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
//   int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );
//
//   cuR_ff( src, dst, dims, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transf_0_12( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r ){
//
//   double* src = REAL( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
//   int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );
//
//   cuR_df( src, dst, dims, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transf_12_0( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   double* dst = REAL( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int* csrc   = ( R_NilValue == csrc_r ) ? NULL : INTEGER( csrc_r );
//   int* cdst   = ( R_NilValue == cdst_r ) ? NULL : INTEGER( cdst_r );
//
//   cuR_fd( src, dst, dims, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// #ifndef CUDA_EXCLUDE
//
// extern "C"
// SEXP cuR_transf_1_3( SEXP src_r, SEXP dst_r, SEXP dims_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyHostToDevice ) );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transf_3_1( SEXP src_r, SEXP dst_r, SEXP dims_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyDeviceToHost ) );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transf_2_3( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP stream_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   if( stream_r != R_NilValue ){
//     cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
//     cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(float), cudaMemcpyHostToDevice, *stream ) );
//
//     // Flush for WDDM
//     cudaStreamQuery(0);
//   }else{
//     cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyHostToDevice ) );
//   }
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transf_3_2( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP stream_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   if( stream_r != R_NilValue ){
//     cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
//     cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(float), cudaMemcpyDeviceToHost, *stream ) );
//
//     // Flush for WDDM
//     cudaStreamQuery(0);
//   }else{
//     cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyDeviceToHost ) );
//   }
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transf_3_3( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP stream_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   if( stream_r != R_NilValue ){
//     cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
//     cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(float), cudaMemcpyDeviceToDevice, *stream ) );
//
//     // Flush for WDDM
//     cudaStreamQuery(0);
//   }else{
//     cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyDeviceToDevice ) );
//     cudaTry( cudaDeviceSynchronize() )
//   }
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// #endif
