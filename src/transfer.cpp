#include "transfer.h"

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

  // Offsets with permutations offset the permutation vector itself
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

  // Out-of-bounds checks only check for permutation content
  // Span checks are done in R

  // Copy
  if( !src_perm_ptr && !src_perm_ptr ){
    // No subsetting
    int l = dims[0]*dims[1];
    for( int j = 0; j < l; j++ ){
      dst_ptr[j] = (d)src_ptr[j];
    }
  }else if( src_perm_ptr && src_perm_ptr ){
    // Both subsetted
    int dst_off, src_off;

    for( int i = 0; i < dims[1]; i ++ ){
      if( src_perm_ptr[i] > src_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on source tensor" );
      }

      if( dst_perm_ptr[i] > dst_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on destination tensor" );
      }

      dst_off = ( src_perm_ptr[i] - 1 ) * dims[0];
      src_off = ( dst_perm_ptr[i] - 1 ) * dims[0];

      for( int j = 0; j < dims[0]; j++ ){
        dst_ptr[dst_off+j] = (d)src_ptr[src_off+j];
      }
    }
  }else if( !src_perm_ptr ){
    // Destination subsetted
    int dst_off;
    int src_off = 0;

    for( int i = 0; i < dims[1]; i ++ ){
      if( dst_perm_ptr[i] > dst_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on destination tensor" );
      }

      dst_off = ( src_perm_ptr[i] - 1 ) * dims[0];

      for( int j = 0; j < dims[0]; j++ ){
        dst_ptr[dst_off+j] = (d)src_ptr[j];
      }

      src_off += dims[0];
    }
  }else{
    // Source subsetted
    int dst_off = 0;
    int src_off;

    for( int i = 0; i < dims[1]; i ++ ){
      if( src_perm_ptr[i] > src_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on source tensor" );
      }

      src_off = ( dst_perm_ptr[i] - 1 ) * dims[0];

      for( int j = 0; j < dims[0]; j++ ){
        dst_ptr[dst_off+j] = (d)src_ptr[src_off+j];
      }

      dst_off += dims[0];
    }
  }

  if( stream_ptr ){
    Rf_warning( "Active stream given to a synchronous transfer call" );
  }
}

template <typename t>
void cuR_transfer_host_device( t* src_ptr,
                               t* dst_ptr,
                               int* src_dims,
                               int* dst_dims,
                               int* dims,
                               int* src_perm_ptr,
                               int* dst_perm_ptr,
                               int src_span_off,
                               int dst_span_off,
                               cudaStream_t* stream_ptr ){

  // Offsets with permutations offset the permutation vector itself
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

  // Out-of-bounds checks only check for permutation content
  // Span checks are done in R

  // Copy
  if( !src_perm_ptr && !src_perm_ptr ){
    // No subsetting
    int l = dims[0]*dims[1];
    if( stream_ptr ){
      cudaTry( cudaMemcpyAsync( dst_ptr,
                                src_ptr,
                                l * sizeof(t),
                                cudaMemcpyHostToDevice,
                                *stream_ptr ) );
    }else{
      cudaTry( cudaMemcpy( dst_ptr,
                           src_ptr,
                           l * sizeof(t),
                           cudaMemcpyHostToDevice ) );
    }
  }else if( src_perm_ptr && src_perm_ptr ){
    // Both subsetted
    int dst_off, src_off;

    for( int i = 0; i < dims[1]; i ++ ){
      if( src_perm_ptr[i] > src_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on source tensor" );
      }

      if( dst_perm_ptr[i] > dst_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on destination tensor" );
      }

      dst_off = ( src_perm_ptr[i] - 1 ) * dims[0];
      src_off = ( dst_perm_ptr[i] - 1 ) * dims[0];

      if( stream_ptr ){
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_off,
                                  src_ptr + src_off,
                                  dims[0] * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpy( dst_ptr + dst_off,
                             src_ptr + src_off,
                             dims[0] * sizeof(t),
                             cudaMemcpyHostToDevice ) );
      }
    }
  }else if( !src_perm_ptr ){
    // Destination subsetted
    int dst_off;
    int src_off = 0;

    for( int i = 0; i < dims[1]; i ++ ){
      if( dst_perm_ptr[i] > dst_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on destination tensor" );
      }

      dst_off = ( src_perm_ptr[i] - 1 ) * dims[0];

      if( stream_ptr ){
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_off,
                                  src_ptr + src_off,
                                  dims[0] * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpy( dst_ptr + dst_off,
                             src_ptr + src_off,
                             dims[0] * sizeof(t),
                             cudaMemcpyHostToDevice ) );
      }

      src_off += dims[0];
    }
  }else{
    // Source subsetted
    int dst_off = 0;
    int src_off;

    for( int i = 0; i < dims[1]; i ++ ){
      if( src_perm_ptr[i] > src_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on source tensor" );
      }

      src_off = ( dst_perm_ptr[i] - 1 ) * dims[0];

      if( stream_ptr ){
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_off,
                                  src_ptr + src_off,
                                  dims[0] * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpy( dst_ptr + dst_off,
                             src_ptr + src_off,
                             dims[0] * sizeof(t),
                             cudaMemcpyHostToDevice ) );
      }

      dst_off += dims[0];
    }
  }

  if( stream_ptr ){
    // Flush for WDDM
    cudaStreamQuery(0);
  }
}

template <typename t>
void cuR_transfer_device_host( t* src_ptr,
                               t* dst_ptr,
                               int* src_dims,
                               int* dst_dims,
                               int* dims,
                               int* src_perm_ptr,
                               int* dst_perm_ptr,
                               int src_span_off,
                               int dst_span_off,
                               cudaStream_t* stream_ptr ){

  // Offsets with permutations offset the permutation vector itself
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

  // Out-of-bounds checks only check for permutation content
  // Span checks are done in R

  // Copy
  if( !src_perm_ptr && !src_perm_ptr ){
    // No subsetting
    int l = dims[0]*dims[1];
    if( stream_ptr ){
      cudaTry( cudaMemcpyAsync( dst_ptr,
                                src_ptr,
                                l * sizeof(t),
                                cudaMemcpyDeviceToHost,
                                *stream_ptr ) );
    }else{
      cudaTry( cudaMemcpy( dst_ptr,
                           src_ptr,
                           l * sizeof(t),
                           cudaMemcpyDeviceToHost ) );
    }
  }else if( src_perm_ptr && src_perm_ptr ){
    // Both subsetted
    int dst_off, src_off;

    for( int i = 0; i < dims[1]; i ++ ){
      if( src_perm_ptr[i] > src_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on source tensor" );
      }

      if( dst_perm_ptr[i] > dst_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on destination tensor" );
      }

      dst_off = ( src_perm_ptr[i] - 1 ) * dims[0];
      src_off = ( dst_perm_ptr[i] - 1 ) * dims[0];

      if( stream_ptr ){
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_off,
                                  src_ptr + src_off,
                                  dims[0] * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpy( dst_ptr + dst_off,
                             src_ptr + src_off,
                             dims[0] * sizeof(t),
                             cudaMemcpyHostToDevice ) );
      }
    }
  }else if( !src_perm_ptr ){
    // Destination subsetted
    int dst_off;
    int src_off = 0;

    for( int i = 0; i < dims[1]; i ++ ){
      if( dst_perm_ptr[i] > dst_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on destination tensor" );
      }

      dst_off = ( src_perm_ptr[i] - 1 ) * dims[0];

      if( stream_ptr ){
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_off,
                                  src_ptr + src_off,
                                  dims[0] * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpy( dst_ptr + dst_off,
                             src_ptr + src_off,
                             dims[0] * sizeof(t),
                             cudaMemcpyHostToDevice ) );
      }

      src_off += dims[0];
    }
  }else{
    // Source subsetted
    int dst_off = 0;
    int src_off;

    for( int i = 0; i < dims[1]; i ++ ){
      if( src_perm_ptr[i] > src_dims[1] ){
        Rf_error( "Out-of-bounds transfer call on source tensor" );
      }

      src_off = ( dst_perm_ptr[i] - 1 ) * dims[0];

      if( stream_ptr ){
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_off,
                                  src_ptr + src_off,
                                  dims[0] * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpy( dst_ptr + dst_off,
                             src_ptr + src_off,
                             dims[0] * sizeof(t),
                             cudaMemcpyHostToDevice ) );
      }

      dst_off += dims[0];
    }
  }

  if( stream_ptr ){
    // Flush for WDDM
    cudaStreamQuery(0);
  }
}

// Main transfer call
// Steel yourself!
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

  // Arg conversions (except src_ptr, dst_ptr)
  int src_level   = Rf_asInteger( src_level_r );
  int dst_level   = Rf_asInteger( dst_level_r );
  const char type = CHAR( STRING_ELT( type_r, 0 ) )[0];
  int* src_dims   = INTEGER( src_dims_r );
  int* dst_dims   = INTEGER( dst_dims_r );
  int* dims       = INTEGER( dims_r );

  int* src_perm_ptr = ( R_NilValue == src_perm_ptr_r ) ? NULL :
    ( TYPEOF( src_perm_ptr_r ) == EXTPTRSXP ? (int*) R_ExternalPtrAddr( src_perm_ptr_r ) :
        INTEGER( src_perm_ptr_r ) );

  int* dst_perm_ptr = ( R_NilValue == dst_perm_ptr_r ) ? NULL :
    ( TYPEOF( dst_perm_ptr_r ) == EXTPTRSXP ? (int*) R_ExternalPtrAddr( dst_perm_ptr_r ) :
        INTEGER( dst_perm_ptr_r ) );

  int src_span_off  = ( R_NilValue == src_span_off_r ) ? 0:
    ( Rf_asInteger( src_span_off_r ) - 1 );

  int dst_span_off  = ( R_NilValue == dst_span_off_r ) ? 0:
    ( Rf_asInteger( dst_span_off_r ) - 1 );

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*) R_ExternalPtrAddr( stream_ptr_r );

  // Calls
  switch( src_level ){
  case 0:
    switch( dst_level ){
    // 0-0 =====================================================================
    case 0:
      switch( type ){
      case 'n':
        cuR_transfer_host_host<double, double>( REAL( src_ptr_r ),
                                                REAL( dst_ptr_r ),
                                                src_dims,
                                                dst_dims,
                                                dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off,
                                                stream_ptr );
        break;

      case 'i':
        cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                          INTEGER( dst_ptr_r ),
                                          src_dims,
                                          dst_dims,
                                          dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        break;

      case 'l':
        cuR_transfer_host_host<int, int>( LOGICAL( src_ptr_r ),
                                          LOGICAL( dst_ptr_r ),
                                          src_dims,
                                          dst_dims,
                                          dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 0-1 -------------------------------------------------------------------
    case 1:
      switch( type ){
      case 'n':
        cuR_transfer_host_host<double, float>( REAL( src_ptr_r ),
                                               (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                               src_dims,
                                               dst_dims,
                                               dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off,
                                               stream_ptr );
        break;

      case 'i':
        cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                          (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                          src_dims,
                                          dst_dims,
                                          dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        break;

      case 'l':
        cuR_transfer_host_host<int, bool>( LOGICAL( src_ptr_r ),
                                           (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                           src_dims,
                                           dst_dims,
                                           dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           stream_ptr );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 0-2 -------------------------------------------------------------------
    case 2:
      switch( type ){
      case 'n':
        cuR_transfer_host_host<double, float>( REAL( src_ptr_r ),
                                               (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                               src_dims,
                                               dst_dims,
                                               dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off,
                                               stream_ptr );
        break;

      case 'i':
        cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                          (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                          src_dims,
                                          dst_dims,
                                          dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        break;

      case 'l':
        cuR_transfer_host_host<int, bool>( LOGICAL( src_ptr_r ),
                                           (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                           src_dims,
                                           dst_dims,
                                           dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           stream_ptr );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

    default:
      Rf_error( "Invalid destination level in transfer call" );
    }
    break;

  case 1:
    switch( dst_level ){
    // 1-0 =====================================================================
    case 0:
      switch( type ){
      case 'n':
        cuR_transfer_host_host<float, double>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                               REAL( dst_ptr_r ),
                                               src_dims,
                                               dst_dims,
                                               dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off,
                                               stream_ptr );
        break;

      case 'i':
        cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                          INTEGER( dst_ptr_r ),
                                          src_dims,
                                          dst_dims,
                                          dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        break;

      case 'l':
        cuR_transfer_host_host<bool, int>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                           LOGICAL( dst_ptr_r ),
                                           src_dims,
                                           dst_dims,
                                           dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           stream_ptr );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 1-1 -------------------------------------------------------------------
    case 1:
      switch( type ){
      case 'n':
        cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                              (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                              src_dims,
                                              dst_dims,
                                              dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              stream_ptr );
        break;

      case 'i':
        cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                          (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                          src_dims,
                                          dst_dims,
                                          dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        break;

      case 'l':
        cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            src_dims,
                                            dst_dims,
                                            dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            stream_ptr );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 1-2 -------------------------------------------------------------------
    case 2:
      switch( type ){
      case 'n':
        cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                              (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                              src_dims,
                                              dst_dims,
                                              dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              stream_ptr );
        break;

      case 'i':
        cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                          (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                          src_dims,
                                          dst_dims,
                                          dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        break;

      case 'l':
        cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            src_dims,
                                            dst_dims,
                                            dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            stream_ptr );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 1-3 -------------------------------------------------------------------
    case 3:
      switch( type ){
      case 'n':
        cuR_transfer_host_device<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                         (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                         src_dims,
                                         dst_dims,
                                         dims,
                                         src_perm_ptr,
                                         dst_perm_ptr,
                                         src_span_off,
                                         dst_span_off,
                                         NULL );
        break;

      case 'i':
        cuR_transfer_host_device<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                       (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                       src_dims,
                                       dst_dims,
                                       dims,
                                       src_perm_ptr,
                                       dst_perm_ptr,
                                       src_span_off,
                                       dst_span_off,
                                       NULL );
        break;

      case 'l':
        cuR_transfer_host_device<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                        (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                        src_dims,
                                        dst_dims,
                                        dims,
                                        src_perm_ptr,
                                        dst_perm_ptr,
                                        src_span_off,
                                        dst_span_off,
                                        NULL );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

    default:
      Rf_error( "Invalid destination level in transfer call" );
    }
    break;

  case 2:
    switch( dst_level ){
    // 2-0 =====================================================================
    case 0:
      switch( type ){
      case 'n':
        cuR_transfer_host_host<float, double>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                               REAL( dst_ptr_r ),
                                               src_dims,
                                               dst_dims,
                                               dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off,
                                               stream_ptr );
        break;

      case 'i':
        cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                          INTEGER( dst_ptr_r ),
                                          src_dims,
                                          dst_dims,
                                          dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        break;

      case 'l':
        cuR_transfer_host_host<bool, int>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                           LOGICAL( dst_ptr_r ),
                                           src_dims,
                                           dst_dims,
                                           dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           stream_ptr );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 2-1 -------------------------------------------------------------------
    case 1:
      switch( type ){
      case 'n':
        cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                              (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                              src_dims,
                                              dst_dims,
                                              dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              stream_ptr );
        break;

      case 'i':
        cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                          (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                          src_dims,
                                          dst_dims,
                                          dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        break;

      case 'l':
        cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            src_dims,
                                            dst_dims,
                                            dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            stream_ptr );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 2-2 -------------------------------------------------------------------
    case 2:
      switch( type ){
      case 'n':
        cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                              (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                              src_dims,
                                              dst_dims,
                                              dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              stream_ptr );
        break;

      case 'i':
        cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                          (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                          src_dims,
                                          dst_dims,
                                          dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        break;

      case 'l':
        cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            src_dims,
                                            dst_dims,
                                            dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            stream_ptr );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 2-3 -------------------------------------------------------------------
    case 3:
      switch( type ){
      case 'n':
        cuR_transfer_host_device<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                         (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                         src_dims,
                                         dst_dims,
                                         dims,
                                         src_perm_ptr,
                                         dst_perm_ptr,
                                         src_span_off,
                                         dst_span_off,
                                         stream_ptr );
        break;

      case 'i':
        cuR_transfer_host_device<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                       (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                       src_dims,
                                       dst_dims,
                                       dims,
                                       src_perm_ptr,
                                       dst_perm_ptr,
                                       src_span_off,
                                       dst_span_off,
                                       stream_ptr );
        break;

      case 'l':
        cuR_transfer_host_device<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                        (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                        src_dims,
                                        dst_dims,
                                        dims,
                                        src_perm_ptr,
                                        dst_perm_ptr,
                                        src_span_off,
                                        dst_span_off,
                                        stream_ptr );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

    default:
      Rf_error( "Invalid destination level in transfer call" );
    }
    break;

  case 3:
    switch( dst_level ){
    // 3-1 =====================================================================
    case 1:
      switch( type ){
      case 'n':
        cuR_transfer_device_host<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                         (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                         src_dims,
                                         dst_dims,
                                         dims,
                                         src_perm_ptr,
                                         dst_perm_ptr,
                                         src_span_off,
                                         dst_span_off,
                                         NULL );
        break;

      case 'i':
        cuR_transfer_device_host<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                       (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                       src_dims,
                                       dst_dims,
                                       dims,
                                       src_perm_ptr,
                                       dst_perm_ptr,
                                       src_span_off,
                                       dst_span_off,
                                       NULL );
        break;

      case 'l':
        cuR_transfer_device_host<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                        (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                        src_dims,
                                        dst_dims,
                                        dims,
                                        src_perm_ptr,
                                        dst_perm_ptr,
                                        src_span_off,
                                        dst_span_off,
                                        NULL );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 3-2 -------------------------------------------------------------------
    case 2:
      switch( type ){
      case 'n':
        cuR_transfer_device_host<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                         (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                         src_dims,
                                         dst_dims,
                                         dims,
                                         src_perm_ptr,
                                         dst_perm_ptr,
                                         src_span_off,
                                         dst_span_off,
                                         stream_ptr );
        break;

      case 'i':
        cuR_transfer_device_host<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                       (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                       src_dims,
                                       dst_dims,
                                       dims,
                                       src_perm_ptr,
                                       dst_perm_ptr,
                                       src_span_off,
                                       dst_span_off,
                                       stream_ptr );
        break;

      case 'l':
        cuR_transfer_device_host<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                        (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                        src_dims,
                                        dst_dims,
                                        dims,
                                        src_perm_ptr,
                                        dst_perm_ptr,
                                        src_span_off,
                                        dst_span_off,
                                        stream_ptr );
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 3-3 -------------------------------------------------------------------
    case 3:
      cuR_transfer_device_device_cu( (void*) R_ExternalPtrAddr( src_ptr_r ),
                                     (void*) R_ExternalPtrAddr( dst_ptr_r ),
                                     type,
                                     src_dims,
                                     dst_dims,
                                     dims,
                                     src_perm_ptr,
                                     dst_perm_ptr,
                                     src_span_off,
                                     dst_span_off,
                                     stream_ptr );
      break;

    default:
      Rf_error( "Invalid destination level in transfer call" );
    }
    break;

  default:
    Rf_error( "Invalid source level in transfer call" );
  }

  return R_NilValue;
}
