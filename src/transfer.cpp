#include "transfer.h"

// template <typename s, typename d>
// void cuR_transfer_host_host( s* src, d* dst, int* dims, int osrc, int odst, int* csrc, int* cdst ){
//   // Offsets now work with column subsetting
//   if( osrc ){
//     if( csrc ){
//       csrc = csrc + osrc;
//     }else{
//       src  = src + (osrc * dims[0]);
//     }
//   }
//
//   if( odst ){
//     if( cdst ){
//       cdst = cdst + odst;
//     }else{
//       dst  = dst + (odst * dims[0]);
//     }
//   }
//
//   // Copy
//   if( !csrc && !cdst ){
//     // No subsetting
//     int l = dims[0]*dims[1];
//     for( int j = 0; j < l; j++ ){
//       dst[j] = (d)src[j];
//     }
//   }else if( csrc && cdst ){
//     // Both subsetted
//     int dst_off, src_off;
//     for( int i = 0; i < dims[1]; i ++ ){
//       dst_off = (cdst[i]-1)*dims[0];
//       src_off = (csrc[i]-1)*dims[0];
//       for( int j = 0; j < dims[0]; j++ ){
//         dst[dst_off+j] = (d)src[src_off+j];
//       }
//     }
//   }else if( !csrc ){
//     // Destination subsetted
//     int dst_off;
//     int src_off = 0;
//     for( int i = 0; i < dims[1]; i ++ ){
//       dst_off = (cdst[i]-1)*dims[0];
//       for( int j = 0; j < dims[0]; j++ ){
//         dst[dst_off+j] = (d)src[src_off+j];
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
//         dst[dst_off+j] = (d)src[src_off+j];
//       }
//       dst_off += dims[0];
//     }
//   }
// }

template <typename s, typename d>
void cuR_transfer_host_host( s* src_ptr,
                             d* dst_ptr,
                             int* src_dims,
                             int* dst_dims,
                             int* dims,
                             int* src_perm_ptr,
                             int* dst_perm_ptr,
                             int src_span_off,
                             int dst_span_off,
                             cudaStream_t* stream_ptr ){

  // Offsets now work with column subsetting
  if( src_span_off ){
    if( src_perm_ptr ){
      src_perm_ptr = src_perm_ptr + src_span_off;
    }else{
      src_ptr = src_ptr + ( src_span_off * dims[0] );
    }
  }

  if( dst_span_off ){
    if( dst_perm_ptr ){
      dst_perm_ptr = dst_perm_ptr + dst_span_off;
    }else{
      dst_ptr = dst_ptr + ( dst_span_off * dims[0] );
    }
  }

  // TODO ====
  // These checks are wrong, not enough

  //Safety maxes to avoid segfaults with perms or offsets
  int src_max = src_dims[0] * src_dims[1] - 1;
  int dst_max = dst_dims[0] * dst_dims[1] - 1;

  // Copy
  if( !src_perm_ptr && !src_perm_ptr ){
    // No subsetting
    int l = dims[0]*dims[1];
    for( int j = 0; j < l; j++ ){
      if( j > src_max ){
        Rf_error( "Out-of-bounds transfer on source tensor" );
      }

      if( j > dst_max ){
        Rf_error( "Out-of-bounds transfer on destination tensor" );
      }

      dst_ptr[j] = (d)src_ptr[j];
    }
  }//else if( csrc && cdst ){
  //   // Both subsetted
  //   int dst_off, src_off, dst_j, src_j;
  //   for( int i = 0; i < dims[1]; i ++ ){
  //     dst_off = (cdst[i]-1)*dims[0];
  //     src_off = (csrc[i]-1)*dims[0];
  //     for( int j = 0; j < dims[0]; j++ ){
  //       dst_j = dst_off+j;
  //       src_j = src_off+j;
  //
  //       if( j > src_max ){
  //         Rf_error( "Out-of-bounds transfer on source tensor" );
  //       }
  //
  //       if( j > dst_max ){
  //         Rf_error( "Out-of-bounds transfer on destination tensor" );
  //       }
  //
  //       dst_ptr[dst_off+j] = (d)src_ptr[src_off+j];
  //     }
  //   }
  // }
  // }else if( !csrc ){
  //   // Destination subsetted
  //   int dst_off;
  //   int src_off = 0;
  //   for( int i = 0; i < dims[1]; i ++ ){
  //     dst_off = (cdst[i]-1)*dims[0];
  //     for( int j = 0; j < dims[0]; j++ ){
  //       dst[dst_off+j] = (d)src[src_off+j];
  //     }
  //     src_off += dims[0];
  //   }
  // }else{
  //   // Source subsetted
  //   int dst_off = 0;
  //   int src_off;
  //   for( int i = 0; i < dims[1]; i ++ ){
  //     src_off = (csrc[i]-1)*dims[0];
  //     for( int j = 0; j < dims[0]; j++ ){
  //       dst[dst_off+j] = (d)src[src_off+j];
  //     }
  //     dst_off += dims[0];
  //   }
  // }
}

// -----------------------------------------------------------------------------
extern "C"
SEXP cuR_transfer( SEXP src_ptr_r,
                   SEXP dst_ptr_r,
                   SEXP src_level_r,
                   SEXP dst_level_r,
                   SEXP type_r,
                   SEXP src_dims_r,
                   SEXP dst_dims_r,
                   SEXP dims_r,
                   SEXP src_perm_ptr_r,  // Optional
                   SEXP dst_perm_ptr_r,  // Optional
                   SEXP src_span_off_r,  // Optional
                   SEXP dst_span_off_r,  // Optional
                   SEXP stream_ptr_r ){  // Optional

  // Arg conversions (except src, dst)
  // void* src_ptr;  // Types change, hence void*
  // void* dst_ptr;
  int src_level   = Rf_asInteger( src_level_r );
  int dst_level   = Rf_asInteger( dst_level_r );
  const char type = CHAR( STRING_ELT( type_r, 0 ) )[0];
  int* src_dims   = INTEGER( src_dims_r );
  int* dst_dims   = INTEGER( dst_dims_r );
  int* dims       = INTEGER( dims_r );

  int* src_perm_ptr = ( R_NilValue == src_perm_ptr_r ) ? NULL :
    ( TYPEOF( src_perm_ptr_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( src_perm_ptr_r ) :
        INTEGER( src_perm_ptr_r ) );

  int* dst_perm_ptr = ( R_NilValue == dst_perm_ptr_r ) ? NULL :
    ( TYPEOF( dst_perm_ptr_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( dst_perm_ptr_r ) :
        INTEGER( dst_perm_ptr_r ) );

  int src_span_off  = ( R_NilValue == src_span_off_r ) ? 0:
    ( Rf_asInteger( src_span_off_r ) - 1);

  int dst_span_off  = ( R_NilValue == dst_span_off_r ) ? 0:
    ( Rf_asInteger( dst_span_off_r ) - 1);

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*)R_ExternalPtrAddr( stream_ptr_r );

  // src, dst conv and calls
  switch( src_level ){
  case 0:
    switch( dst_level ){
    case 0:
      switch( type ){
      case 'n':
        double* src_ptr = REAL( src_ptr_r );
        double* dst_ptr = REAL( dst_ptr_r );
        cuR_transfer_host_host<double, double>( src_ptr,
                                                dst_ptr,
                                                src_dims,
                                                dst_dims,
                                                dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off,
                                                stream_ptr );
        break;
      }
    }
  }

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( TRUE ) );
  Rf_unprotect(1);
  return ret_r;
}


// extern "C"
// SEXP cuR_transfer_0_0_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   double* src = REAL( src_r );
//   double* dst = REAL( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//
//   cuR_transfer_host_host<double, double>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_0_0_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   int* src    = INTEGER( src_r );
//   int* dst    = INTEGER( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//
//   cuR_transfer_host_host<int, int>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_0_0_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   int* src    = LOGICAL( src_r );
//   int* dst    = LOGICAL( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//   cuR_transfer_host_host<int, int>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_0_12_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   double* src = REAL( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//
//   cuR_transfer_host_host<double, float>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_0_12_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   int* src    = INTEGER( src_r );
//   int* dst    = (int*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//
//   cuR_transfer_host_host<int, int>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_0_12_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   int* src    = LOGICAL( src_r );
//   bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//
//   cuR_transfer_host_host<int, bool>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_12_0_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   double* dst = REAL( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//
//   cuR_transfer_host_host<float, double>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_12_0_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   int* src    = (int*)R_ExternalPtrAddr( src_r );
//   int* dst    = INTEGER( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//
//   cuR_transfer_host_host<int, int>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_12_0_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   bool* src   = (bool*)R_ExternalPtrAddr( src_r );
//   int* dst    = LOGICAL( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//
//   cuR_transfer_host_host<bool, int>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_12_12_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = R_NilValue == osrc_r ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = R_NilValue == odst_r ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//
//   cuR_transfer_host_host<float, float>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_12_12_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   int* src    = (int*)R_ExternalPtrAddr( src_r );
//   int* dst    = (int*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//
//   cuR_transfer_host_host<int, int>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_12_12_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r ){
//
//   bool* src   = (bool*)R_ExternalPtrAddr( src_r );
//   bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = R_NilValue == csrc_r ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = R_NilValue == cdst_r ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//
//   cuR_transfer_host_host<bool, bool>( src, dst, dims, osrc, odst, csrc, cdst );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// #ifndef CUDA_EXCLUDE
//
// extern "C"
// SEXP cuR_transfer_1_3_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
//
//   cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyHostToDevice ) );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_1_3_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){
//
//   int* src    = (int*)R_ExternalPtrAddr( src_r );
//   int* dst    = (int*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
//
//   cudaTry( cudaMemcpy( dst, src, l*sizeof(int), cudaMemcpyHostToDevice ) );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_1_3_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){
//
//   bool* src   = (bool*)R_ExternalPtrAddr( src_r );
//   bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
//
//   cudaTry( cudaMemcpy( dst, src, l*sizeof(bool), cudaMemcpyHostToDevice ) );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_3_1_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
//
//   cudaTry( cudaMemcpy( dst, src, l*sizeof(float), cudaMemcpyDeviceToHost ) );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_3_1_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){
//
//   int* src    = (int*)R_ExternalPtrAddr( src_r );
//   int* dst    = (int*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
//
//   cudaTry( cudaMemcpy( dst, src, l*sizeof(int), cudaMemcpyDeviceToHost ) );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_3_1_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r ){
//
//   bool* src   = (bool*)R_ExternalPtrAddr( src_r );
//   bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
//
//   cudaTry( cudaMemcpy( dst, src, l*sizeof(bool), cudaMemcpyDeviceToHost ) );
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_2_3_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
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
// SEXP cuR_transfer_2_3_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){
//
//   int* src    = (int*)R_ExternalPtrAddr( src_r );
//   int* dst    = (int*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
//
//   if( stream_r != R_NilValue ){
//     cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
//     cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(int), cudaMemcpyHostToDevice, *stream ) );
//
//     // Flush for WDDM
//     cudaStreamQuery(0);
//   }else{
//     cudaTry( cudaMemcpy( dst, src, l*sizeof(int), cudaMemcpyHostToDevice ) );
//   }
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_2_3_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){
//
//   bool* src   = (bool*)R_ExternalPtrAddr( src_r );
//   bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
//
//   if( stream_r != R_NilValue ){
//     cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
//     cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(bool), cudaMemcpyHostToDevice, *stream ) );
//
//     // Flush for WDDM
//     cudaStreamQuery(0);
//   }else{
//     cudaTry( cudaMemcpy( dst, src, l*sizeof(bool), cudaMemcpyHostToDevice ) );
//   }
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_3_2_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
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
// SEXP cuR_transfer_3_2_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){
//
//   int* src    = (int*)R_ExternalPtrAddr( src_r );
//   int* dst    = (int*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
//
//   if( stream_r != R_NilValue ){
//     cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
//     cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(int), cudaMemcpyDeviceToHost, *stream ) );
//
//     // Flush for WDDM
//     cudaStreamQuery(0);
//   }else{
//     cudaTry( cudaMemcpy( dst, src, l*sizeof(int), cudaMemcpyDeviceToHost ) );
//   }
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_3_2_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){
//
//   bool* src   = (bool*)R_ExternalPtrAddr( src_r );
//   bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* dims   = INTEGER( dims_r );
//   int l       = dims[0]*dims[1];
//
//   src  = src + (osrc * dims[0]);
//   dst  = dst + (odst * dims[0]);
//
//   if( stream_r != R_NilValue ){
//     cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
//     cudaTry( cudaMemcpyAsync( dst, src, l*sizeof(bool), cudaMemcpyDeviceToHost, *stream ) );
//
//     // Flush for WDDM
//     cudaStreamQuery(0);
//   }else{
//     cudaTry( cudaMemcpy( dst, src, l*sizeof(bool), cudaMemcpyDeviceToHost ) );
//   }
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_3_3_n( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){
//
//   float* src  = (float*)R_ExternalPtrAddr( src_r );
//   float* dst  = (float*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = ( R_NilValue == csrc_r ) ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = ( R_NilValue == cdst_r ) ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//   cudaStream_t* stream = ( R_NilValue == stream_r ) ? NULL : (cudaStream_t*)R_ExternalPtrAddr( stream_r );
//
//   cuR_transfer_device_device_n_cu( src, dst, dims, osrc, odst, csrc, cdst, stream );
//
//   if( stream_r != R_NilValue ){
//     // Flush for WDDM
//     cudaStreamQuery(0);
//   }else{
//     cudaTry( cudaDeviceSynchronize() );
//   }
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_3_3_i( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){
//
//   int* src    = (int*)R_ExternalPtrAddr( src_r );
//   int* dst    = (int*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = ( R_NilValue == csrc_r ) ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = ( R_NilValue == cdst_r ) ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//   cudaStream_t* stream = ( R_NilValue == stream_r ) ? NULL : (cudaStream_t*)R_ExternalPtrAddr( stream_r );
//
//   cuR_transfer_device_device_i_cu( src, dst, dims, osrc, odst, csrc, cdst, stream );
//
//   if( stream_r != R_NilValue ){
//     // Flush for WDDM
//     cudaStreamQuery(0);
//   }else{
//     cudaTry( cudaDeviceSynchronize() )
//   }
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuR_transfer_3_3_l( SEXP src_r, SEXP dst_r, SEXP dims_r, SEXP csrc_r, SEXP cdst_r, SEXP osrc_r, SEXP odst_r, SEXP stream_r ){
//
//   bool* src   = (bool*)R_ExternalPtrAddr( src_r );
//   bool* dst   = (bool*)R_ExternalPtrAddr( dst_r );
//   int* dims   = INTEGER( dims_r );
//   int osrc    = ( R_NilValue == osrc_r ) ? 0 : (Rf_asInteger( osrc_r ) - 1);
//   int odst    = ( R_NilValue == odst_r ) ? 0 : (Rf_asInteger( odst_r ) - 1);
//   int* csrc   = ( R_NilValue == csrc_r ) ? NULL :
//     ( TYPEOF( csrc_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( csrc_r ) : INTEGER( csrc_r ) );
//   int* cdst   = ( R_NilValue == cdst_r ) ? NULL :
//     ( TYPEOF( cdst_r ) == EXTPTRSXP ? (int*)R_ExternalPtrAddr( cdst_r ) : INTEGER( cdst_r ) );
//   cudaStream_t* stream = ( R_NilValue == stream_r ) ? NULL : (cudaStream_t*)R_ExternalPtrAddr( stream_r );
//
//   cuR_transfer_device_device_l_cu( src, dst, dims, osrc, odst, csrc, cdst, stream );
//
//   if( stream_r != R_NilValue ){
//     // Flush for WDDM
//     cudaStreamQuery(0);
//   }else{
//     cudaTry( cudaDeviceSynchronize() )
//   }
//
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// #endif
