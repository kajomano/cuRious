#include "common_R.h"
#include "common_debug.h"
#include "transfer.h"  // Includes common_cuda.h
#include "streams.h"
#include "omp.h"

#include <vector>
#include <algorithm>

template <typename s, typename d>
void cuR_transfer_host_host( s* src_ptr,
                             d* dst_ptr,
                             int* dims,
                             int* src_dims,
                             int* dst_dims,
                             int* src_perm_ptr,
                             int* dst_perm_ptr,
                             int src_span_off,
                             int dst_span_off ){

  int src_dims_1;
  if( src_dims ){
    src_dims_1 = src_dims[1];
  }else{
    src_dims_1 = 0;
  }

  int dst_dims_1;
  if( dst_dims ){
    dst_dims_1 = dst_dims[1];
  }else{
    dst_dims_1 = 0;
  }

  int dims_0 = dims[0];
  int dims_1 = dims[1];

  // Offsets with permutations offset the permutation vector itself
  if( src_span_off ){
    if( src_perm_ptr ){
      src_perm_ptr = src_perm_ptr + src_span_off;
    }else{
      src_ptr = src_ptr + ( src_span_off * dims_0 );
    }
  }

  if( dst_span_off ){
    if( dst_perm_ptr ){
      dst_perm_ptr = dst_perm_ptr + dst_span_off;
    }else{
      dst_ptr = dst_ptr + ( dst_span_off * dims_0 );
    }
  }

  int dst_pos;
  int src_pos;

  // Worker thread pool
  // int num_workers = std::min(
  //   std::max(
  //     (int)std::thread::hardware_concurrency() / 4,
  //     1
  //   ),
  //   dims_1
  // );

  int num_workers = 6;

  std::vector <std::thread> workers( num_workers - 1 );

  int span_workers = dims_1 / num_workers;
  int rest_workers = 0;

  // Out-of-bounds checks only check for permutation content
  // Span checks are done in R

  // Copy
  if( !src_perm_ptr && !dst_perm_ptr ){
    // No subsetting
    for( int worker = 0; worker < num_workers - 1; worker++ ){
      workers[worker] = std::thread( [=] {
    //
    //     int dst_pos;
    //     int src_pos;
    //
    //     for( int i = rest_workers; i < rest_workers + span_workers; i++ ){
    //       dst_pos = i * dims_0;
    //       src_pos = i * dims_0;
    //
    //       for( int j = 0; j < dims_0; j++ ){
    //         dst_ptr[dst_pos + j] = (d)src_ptr[src_pos + j];
    //       }
    //     }
      });

      // rest_workers += span_workers;
    }

    // for( int i = rest_workers; i < dims_1; i++ ){
    //   dst_pos = i * dims_0;
    //   src_pos = i * dims_0;
    //
    //   for( int j = 0; j < dims_0; j++ ){
    //     dst_ptr[dst_pos + j] = (d)src_ptr[src_pos + j];
    //   }
    // }

    for( auto& w : workers ){
      w.join();
    }
  }
  else if( src_perm_ptr && dst_perm_ptr ){
    // Both subsetted
    for( int worker = 0; worker < num_workers - 1; worker++ ){
      workers[worker] = std::thread( [=] {

        int dst_pos;
        int src_pos;

        for( int i = rest_workers; i < rest_workers + span_workers; i++ ){
          if( src_perm_ptr[i] > src_dims_1 ){
            continue;
          }

          if( dst_perm_ptr[i] > dst_dims_1 ){
            continue;
          }

          src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
          dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

          for( int j = 0; j < dims_0; j++ ){
            dst_ptr[dst_pos + j] = (d)src_ptr[src_pos + j];
          }
        }
      });

      rest_workers += span_workers;
    }

    for( int i = rest_workers; i < dims_1; i++ ){
      if( src_perm_ptr[i] > src_dims_1 ){
        continue;
      }

      if( dst_perm_ptr[i] > dst_dims_1 ){
        continue;
      }

      src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
      dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

      for( int j = 0; j < dims_0; j++ ){
        dst_ptr[dst_pos + j] = (d)src_ptr[src_pos + j];
      }
    }

    for( auto& w : workers ){
      w.join();
    }
  }
  else if( dst_perm_ptr ){
    // Destination subsetted
    for( int worker = 0; worker < num_workers - 1; worker++ ){
      workers[worker] = std::thread( [=] {

        int dst_pos;
        int src_pos;

        for( int i = rest_workers; i < rest_workers + span_workers; i++ ){
          if( dst_perm_ptr[i] > dst_dims_1 ){
            continue;
          }

          src_pos = i * dims_0;
          dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

          for( int j = 0; j < dims_0; j++ ){
            dst_ptr[dst_pos + j] = (d)src_ptr[src_pos + j];
          }
        }
      });

      rest_workers += span_workers;
    }

    for( int i = rest_workers; i < dims_1; i++ ){
      if( dst_perm_ptr[i] > dst_dims_1 ){
        continue;
      }

      src_pos = i * dims_0;
      dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

      for( int j = 0; j < dims_0; j++ ){
        dst_ptr[dst_pos + j] = (d)src_ptr[src_pos + j];
      }
    }

    for( auto& w : workers ){
      w.join();
    }
  }
  else{
    // Source subsetted
    for( int worker = 0; worker < num_workers - 1; worker++ ){
      workers[worker] = std::thread( [=] {

        int dst_pos;
        int src_pos;

        for( int i = rest_workers; i < rest_workers + span_workers; i++ ){
          if( src_perm_ptr[i] > src_dims_1 ){
            continue;
          }

          src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
          dst_pos = i * dims_0;

          for( int j = 0; j < dims_0; j++ ){
            dst_ptr[dst_pos+j] = (d)src_ptr[src_pos+j];
          }
        }
      });

      rest_workers += span_workers;
    }

    for( int i = rest_workers; i < dims_1; i++ ){
      if( src_perm_ptr[i] > src_dims_1 ){
        continue;
      }

      src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
      dst_pos = i * dims_0;

      for( int j = 0; j < dims_0; j++ ){
        dst_ptr[dst_pos+j] = (d)src_ptr[src_pos+j];
      }
    }

    for( auto& w : workers ){
      w.join();
    }
  }
}

#ifndef CUDA_EXCLUDE

template <typename t>
void cuR_transfer_host_device( t* src_ptr,
                               t* dst_ptr,
                               int* dims,
                               int* src_dims,
                               int* dst_dims,
                               int* src_perm_ptr,
                               int* dst_perm_ptr,
                               int src_span_off,
                               int dst_span_off,
                               cudaStream_t* stream_ptr ){

  int src_dims_1;
  if( src_dims ){
    src_dims_1 = src_dims[1];
  }else{
    src_dims_1 = 0;
  }

  int dst_dims_1;
  if( dst_dims ){
    dst_dims_1 = dst_dims[1];
  }else{
    dst_dims_1 = 0;
  }

  int dims_0 = dims[0];
  int dims_1 = dims[1];

  // Offsets with permutations offset the permutation vector itself
  if( src_span_off ){
    if( src_perm_ptr ){
      src_perm_ptr = src_perm_ptr + src_span_off;
    }else{
      src_ptr = src_ptr + ( src_span_off * dims_0 );
    }
  }

  if( dst_span_off ){
    if( dst_perm_ptr ){
      dst_perm_ptr = dst_perm_ptr + dst_span_off;
    }else{
      dst_ptr = dst_ptr + ( dst_span_off * dims_0 );
    }
  }

  int src_pos;
  int dst_pos;

  // Copy
  if( !src_perm_ptr && !dst_perm_ptr ){
    // No subsetting
    if( stream_ptr ){
      cudaTry( cudaMemcpyAsync( dst_ptr,
                                src_ptr,
                                sizeof(t) * dims_0 * dims_1,
                                cudaMemcpyHostToDevice,
                                *stream_ptr ) );
    }else{
      cudaTry( cudaMemcpyAsync( dst_ptr,
                                src_ptr,
                                sizeof(t) * dims_0 * dims_1,
                                cudaMemcpyHostToDevice ) );
    }
  }else if( src_perm_ptr && dst_perm_ptr ){
    for( int i = 0; i < dims_1; i ++ ){
      if( src_perm_ptr[i] > src_dims_1 ){
        continue;
      }

      if( dst_perm_ptr[i] > dst_dims_1 ){
        continue;
      }

      src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
      dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

      if( stream_ptr ){
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyHostToDevice ) );
      }
    }
  }else if( dst_perm_ptr ){
    for( int i = 0; i < dims_1; i ++ ){
      if( dst_perm_ptr[i] > dst_dims_1 ){
        continue;
      }

      src_pos = i * dims_0;
      dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

      if( stream_ptr ){
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyHostToDevice ) );
      }
    }
  }else{
    for( int i = 0; i < dims_1; i ++ ){
      if( src_perm_ptr[i] > src_dims_1 ){
        continue;
      }

      src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
      dst_pos = i * dims_0;

      if( stream_ptr ){
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyHostToDevice ) );
      }
    }
  }

  if( stream_ptr ){
    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaDeviceSynchronize();
  }
}

template <typename t>
void cuR_transfer_device_host( t* src_ptr,
                               t* dst_ptr,
                               int* dims,
                               int* src_dims,
                               int* dst_dims,
                               int* src_perm_ptr,
                               int* dst_perm_ptr,
                               int src_span_off,
                               int dst_span_off,
                               cudaStream_t* stream_ptr ){

  int src_dims_1;
  if( src_dims ){
    src_dims_1 = src_dims[1];
  }else{
    src_dims_1 = 0;
  }

  int dst_dims_1;
  if( dst_dims ){
    dst_dims_1 = dst_dims[1];
  }else{
    dst_dims_1 = 0;
  }

  int dims_0 = dims[0];
  int dims_1 = dims[1];

  // Offsets with permutations offset the permutation vector itself
  if( src_span_off ){
    if( src_perm_ptr ){
      src_perm_ptr = src_perm_ptr + src_span_off;
    }else{
      src_ptr = src_ptr + ( src_span_off * dims_0 );
    }
  }

  if( dst_span_off ){
    if( dst_perm_ptr ){
      dst_perm_ptr = dst_perm_ptr + dst_span_off;
    }else{
      dst_ptr = dst_ptr + ( dst_span_off * dims_0 );
    }
  }

  int dst_pos;
  int src_pos;

  // Copy
  if( !src_perm_ptr && !dst_perm_ptr ){
    // No subsetting
    if( stream_ptr ){
      cudaTry( cudaMemcpyAsync( dst_ptr,
                                src_ptr,
                                sizeof(t) * dims_0 * dims_1,
                                cudaMemcpyDeviceToHost,
                                *stream_ptr ) );
    }else{
      cudaTry( cudaMemcpyAsync( dst_ptr,
                                src_ptr,
                                sizeof(t) * dims_0 * dims_1,
                                cudaMemcpyDeviceToHost ) );
    }
  }else if( src_perm_ptr && dst_perm_ptr ){
    for( int i = 0; i < dims_1; i ++ ){
      if( src_perm_ptr[i] > src_dims_1 ){
        continue;
      }

      if( dst_perm_ptr[i] > dst_dims_1 ){
        continue;
      }

      src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
      dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

      if( stream_ptr ){
        // This copy is not async to the host if dst is unpinned, or just takes
        // a very long time.
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyDeviceToHost,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyDeviceToHost ) );
      }
    }
  }else if( dst_perm_ptr ){
    for( int i = 0; i < dims_1; i ++ ){
      if( dst_perm_ptr[i] > dst_dims_1 ){
        continue;
      }

      src_pos = i * dims_0;
      dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

      if( stream_ptr ){
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyDeviceToHost,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyDeviceToHost ) );
      }
    }
  }else{
    for( int i = 0; i < dims_1; i ++ ){
      if( src_perm_ptr[i] > src_dims_1 ){
        continue;
      }

      src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
      dst_pos = i * dims_0;

      if( stream_ptr ){
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyDeviceToHost,
                                  *stream_ptr ) );
      }else{
        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyDeviceToHost ) );
      }
    }
  }

  if( stream_ptr ){
    // Flush for WDDM
    cudaStreamQuery(0);
  }else{
    cudaDeviceSynchronize();
  }
}

#endif

// Main transfer call
// Steel yourself!
// -----------------------------------------------------------------------------
extern "C"
SEXP cuR_transfer( SEXP src_ptr_r,
                   SEXP dst_ptr_r,
                   SEXP src_level_r,
                   SEXP dst_level_r,
                   SEXP type_r,
                   SEXP dims_r,
                   SEXP src_dims_r,      // Optional*
                   SEXP dst_dims_r,      // Optional**
                   SEXP src_perm_ptr_r,  // Optional*
                   SEXP dst_perm_ptr_r,  // Optional**
                   SEXP src_span_off_r,  // Optional
                   SEXP dst_span_off_r,  // Optional
                   SEXP queue_ptr_r,     // Optional
                   SEXP stream_ptr_r ){  // Optional

  // Arg conversions (except src_ptr, dst_ptr)
  int src_level     = Rf_asInteger( src_level_r );
  int dst_level     = Rf_asInteger( dst_level_r );
  const char type   = CHAR( STRING_ELT( type_r, 0 ) )[0];
  int* dims         = INTEGER( dims_r );

  int* src_dims     = ( R_NilValue == src_dims_r ) ? NULL : INTEGER( src_dims_r );
  int* dst_dims     = ( R_NilValue == dst_dims_r ) ? NULL : INTEGER( dst_dims_r );

  int* src_perm_ptr = ( R_NilValue == src_perm_ptr_r ) ? NULL :
    ( TYPEOF( src_perm_ptr_r ) == EXTPTRSXP ? (int*) R_ExternalPtrAddr( src_perm_ptr_r ) :
        INTEGER( src_perm_ptr_r ) );

  int* dst_perm_ptr = ( R_NilValue == dst_perm_ptr_r ) ? NULL :
    ( TYPEOF( dst_perm_ptr_r ) == EXTPTRSXP ? (int*) R_ExternalPtrAddr( dst_perm_ptr_r ) :
        INTEGER( dst_perm_ptr_r ) );

  int src_span_off  = ( R_NilValue == src_span_off_r ) ? 0 :
    Rf_asInteger( src_span_off_r ) - 1;
  int dst_span_off  = ( R_NilValue == dst_span_off_r ) ? 0 :
    Rf_asInteger( dst_span_off_r ) - 1;

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*) R_ExternalPtrAddr( stream_ptr_r );

  sd_queue* queue_ptr = ( R_NilValue == queue_ptr_r ) ? NULL :
    (sd_queue*) R_ExternalPtrAddr( queue_ptr_r );

  if( src_perm_ptr && !src_dims ){
    Rf_error( "Source dimensions need to be supplied with permutation" );
  }

  if( dst_perm_ptr && !dst_dims ){
    Rf_error( "Destination dimensions need to be supplied with permutation" );
  }

  // Calls
  switch( src_level ){
  case 0:
    switch( dst_level ){
    // 0-0 =====================================================================
    case 0:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<double, double>( REAL( src_ptr_r ),
                                                    REAL( dst_ptr_r ),
                                                    dims,
                                                    src_dims,
                                                    dst_dims,
                                                    src_perm_ptr,
                                                    dst_perm_ptr,
                                                    src_span_off,
                                                    dst_span_off );
          });
        }else{
          cuR_transfer_host_host<double, double>( REAL( src_ptr_r ),
                                                  REAL( dst_ptr_r ),
                                                  dims,
                                                  src_dims,
                                                  dst_dims,
                                                  src_perm_ptr,
                                                  dst_perm_ptr,
                                                  src_span_off,
                                                  dst_span_off );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                              INTEGER( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                            INTEGER( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, int>( LOGICAL( src_ptr_r ),
                                              LOGICAL( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, int>( LOGICAL( src_ptr_r ),
                                            LOGICAL( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 0-1 -------------------------------------------------------------------
    case 1:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<double, float>( REAL( src_ptr_r ),
                                                   (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                   dims,
                                                   src_dims,
                                                   dst_dims,
                                                   src_perm_ptr,
                                                   dst_perm_ptr,
                                                   src_span_off,
                                                   dst_span_off );
          });
        }else{
          cuR_transfer_host_host<double, float>( REAL( src_ptr_r ),
                                                 (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                 dims,
                                                 src_dims,
                                                 dst_dims,
                                                 src_perm_ptr,
                                                 dst_perm_ptr,
                                                 src_span_off,
                                                 dst_span_off );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                            (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, bool>( LOGICAL( src_ptr_r ),
                                               (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                               dims,
                                               src_dims,
                                               dst_dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, bool>( LOGICAL( src_ptr_r ),
                                             (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

#ifndef CUDA_EXCLUDE

      // 0-2 -------------------------------------------------------------------
    case 2:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<double, float>( REAL( src_ptr_r ),
                                                   (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                   dims,
                                                   src_dims,
                                                   dst_dims,
                                                   src_perm_ptr,
                                                   dst_perm_ptr,
                                                   src_span_off,
                                                   dst_span_off );
          });
        }else{
          cuR_transfer_host_host<double, float>( REAL( src_ptr_r ),
                                                 (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                 dims,
                                                 src_dims,
                                                 dst_dims,
                                                 src_perm_ptr,
                                                 dst_perm_ptr,
                                                 src_span_off,
                                                 dst_span_off );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                            (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, bool>( LOGICAL( src_ptr_r ),
                                               (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                               dims,
                                               src_dims,
                                               dst_dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, bool>( LOGICAL( src_ptr_r ),
                                             (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

    default:
      Rf_error( "Invalid destination level in transfer call" );
    }
    break;

#endif

  case 1:
    switch( dst_level ){
    // 1-0 =====================================================================
    case 0:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<float, double>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                   REAL( dst_ptr_r ),
                                                   dims,
                                                   src_dims,
                                                   dst_dims,
                                                   src_perm_ptr,
                                                   dst_perm_ptr,
                                                   src_span_off,
                                                   dst_span_off );
          });
        }else{
          cuR_transfer_host_host<float, double>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                 REAL( dst_ptr_r ),
                                                 dims,
                                                 src_dims,
                                                 dst_dims,
                                                 src_perm_ptr,
                                                 dst_perm_ptr,
                                                 src_span_off,
                                                 dst_span_off );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              INTEGER( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                            INTEGER( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<bool, int>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                               LOGICAL( dst_ptr_r ),
                                               dims,
                                               src_dims,
                                               dst_dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off );
          });
        }else{
          cuR_transfer_host_host<bool, int>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                             LOGICAL( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 1-1 -------------------------------------------------------------------
    case 1:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                  (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                  dims,
                                                  src_dims,
                                                  dst_dims,
                                                  src_perm_ptr,
                                                  dst_perm_ptr,
                                                  src_span_off,
                                                  dst_span_off );
          });
        }else{
          cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                            (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                                (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off );
          });
        }else{
          cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                              (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

#ifndef CUDA_EXCLUDE

      // 1-2 -------------------------------------------------------------------
    case 2:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                  (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                  dims,
                                                  src_dims,
                                                  dst_dims,
                                                  src_perm_ptr,
                                                  dst_perm_ptr,
                                                  src_span_off,
                                                  dst_span_off );
          });
        }else{
          cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                            (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                                (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off );
          });
        }else{
          cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                              (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 1-3 -------------------------------------------------------------------
    case 3:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_device<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                             (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off,
                                             NULL );
          });
        }else{
          cuR_transfer_host_device<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                           (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           NULL );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_device<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                           (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           NULL );
          });
        }else{
          cuR_transfer_host_device<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                         (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                         dims,
                                         src_dims,
                                         dst_dims,
                                         src_perm_ptr,
                                         dst_perm_ptr,
                                         src_span_off,
                                         dst_span_off,
                                         NULL );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_device<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            NULL );
          });
        }else{
          cuR_transfer_host_device<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                          (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                          dims,
                                          src_dims,
                                          dst_dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          NULL );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

#endif

    default:
      Rf_error( "Invalid destination level in transfer call" );
    }
    break;

#ifndef CUDA_EXCLUDE

  case 2:
    switch( dst_level ){
    // 2-0 =====================================================================
    case 0:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<float, double>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                   REAL( dst_ptr_r ),
                                                   dims,
                                                   src_dims,
                                                   dst_dims,
                                                   src_perm_ptr,
                                                   dst_perm_ptr,
                                                   src_span_off,
                                                   dst_span_off );
          });
        }else{
          cuR_transfer_host_host<float, double>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                 REAL( dst_ptr_r ),
                                                 dims,
                                                 src_dims,
                                                 dst_dims,
                                                 src_perm_ptr,
                                                 dst_perm_ptr,
                                                 src_span_off,
                                                 dst_span_off );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              INTEGER( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                            INTEGER( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<bool, int>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                               LOGICAL( dst_ptr_r ),
                                               dims,
                                               src_dims,
                                               dst_dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off );
          });
        }else{
          cuR_transfer_host_host<bool, int>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                             LOGICAL( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 2-1 -------------------------------------------------------------------
    case 1:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                  (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                  dims,
                                                  src_dims,
                                                  dst_dims,
                                                  src_perm_ptr,
                                                  dst_perm_ptr,
                                                  src_span_off,
                                                  dst_span_off );
          });
        }else{
          cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                            (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                                (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off );
          });
        }else{
          cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                              (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 2-2 -------------------------------------------------------------------
    case 2:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                  (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                  dims,
                                                  src_dims,
                                                  dst_dims,
                                                  src_perm_ptr,
                                                  dst_perm_ptr,
                                                  src_span_off,
                                                  dst_span_off );
          });
        }else{
          cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
          });
        }else{
          cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                            (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                                (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off );
          });
        }else{
          cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                              (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 2-3 -------------------------------------------------------------------
    case 3:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_device<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                             (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off,
                                             stream_ptr );
          });
        }else{
          cuR_transfer_host_device<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                           (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           stream_ptr );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_device<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                           (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           stream_ptr );
          });
        }else{
          cuR_transfer_host_device<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                         (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                         dims,
                                         src_dims,
                                         dst_dims,
                                         src_perm_ptr,
                                         dst_perm_ptr,
                                         src_span_off,
                                         dst_span_off,
                                         stream_ptr );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_host_device<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            stream_ptr );
          });
        }else{
          cuR_transfer_host_device<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                          (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                          dims,
                                          src_dims,
                                          dst_dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        }
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
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_device_host<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                             (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off,
                                             NULL );
          });
        }else{
          cuR_transfer_device_host<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                           (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           NULL );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_device_host<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                           (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           NULL );
          });
        }else{
          cuR_transfer_device_host<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                         (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                         dims,
                                         src_dims,
                                         dst_dims,
                                         src_perm_ptr,
                                         dst_perm_ptr,
                                         src_span_off,
                                         dst_span_off,
                                         NULL );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_device_host<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            NULL );
          });
        }else{
          cuR_transfer_device_host<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                          (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                          dims,
                                          src_dims,
                                          dst_dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          NULL );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 3-2 -------------------------------------------------------------------
    case 2:
      switch( type ){
      case 'n':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_device_host<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                             (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off,
                                             stream_ptr );
          });
        }else{
          cuR_transfer_device_host<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                           (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           stream_ptr );
        }
        break;

      case 'i':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_device_host<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                           (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           stream_ptr );
          });
        }else{
          cuR_transfer_device_host<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                         (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                         dims,
                                         src_dims,
                                         dst_dims,
                                         src_perm_ptr,
                                         dst_perm_ptr,
                                         src_span_off,
                                         dst_span_off,
                                         stream_ptr );
        }
        break;

      case 'l':
        if( queue_ptr ){
          queue_ptr -> dispatch( [=]{
            cuR_transfer_device_host<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            stream_ptr );
          });
        }else{
          cuR_transfer_device_host<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                          (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                          dims,
                                          src_dims,
                                          dst_dims,
                                          src_perm_ptr,
                                          dst_perm_ptr,
                                          src_span_off,
                                          dst_span_off,
                                          stream_ptr );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 3-3 -------------------------------------------------------------------
    case 3:
      if( queue_ptr ){
        queue_ptr -> dispatch( [=]{
          cuR_transfer_device_device_cu( (void*) R_ExternalPtrAddr( src_ptr_r ),
                                         (void*) R_ExternalPtrAddr( dst_ptr_r ),
                                         type,
                                         dims,
                                         src_dims,
                                         dst_dims,
                                         src_perm_ptr,
                                         dst_perm_ptr,
                                         src_span_off,
                                         dst_span_off,
                                         stream_ptr );
        });
      }else{
        cuR_transfer_device_device_cu( (void*) R_ExternalPtrAddr( src_ptr_r ),
                                       (void*) R_ExternalPtrAddr( dst_ptr_r ),
                                       type,
                                       dims,
                                       src_dims,
                                       dst_dims,
                                       src_perm_ptr,
                                       dst_perm_ptr,
                                       src_span_off,
                                       dst_span_off,
                                       stream_ptr );
      }
      break;

    default:
      Rf_error( "Invalid destination level in transfer call" );
    }
    break;

#endif

  default:
    Rf_error( "Invalid source level in transfer call" );
  }

  return R_NilValue;
}
