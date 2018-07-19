#include "common_R.h"
#include "common_debug.h"
#include "common_cuda.h"

#include "transfer.h"
#include "streams.h"

#include <typeinfo>
#include <cstring>

// Very simple logic for now
int cuR_transfer_task_span( int dims_0, int dims_1, sd_queue* worker_q_ptr ){
  const int thread_split = 2;
  int threads = (int)worker_q_ptr -> thread_cnt();

  if( ( threads * thread_split ) > dims_1 ){
    return 1;
  }else{
    return dims_1 / ( threads * thread_split );
  }
}

template <typename s, typename d>
void cuR_transfer_host_host( s* src_ptr,
                             d* dst_ptr,
                             int* dims,
                             int* src_dims,
                             int* dst_dims,
                             int* src_perm_ptr,
                             int* dst_perm_ptr,
                             int src_span_off,
                             int dst_span_off,
                             sd_queue* worker_q_ptr = NULL,
                             void* stream_ptr       = NULL ){

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

  sd_queue* worker_q = ( !worker_q_ptr ) ? new sd_queue( 4, false ) : worker_q_ptr;

  int span_task = cuR_transfer_task_span( dims_0, dims_1, worker_q );
  int task = 0;

  // Out-of-bounds checks only check for permutation content
  // Span checks are done in R

#ifndef CUDA_EXCLUDE
  cudaStream_t* stream_ = (cudaStream_t*) stream_ptr;
  if( stream_ ){
    cudaTry( cudaStreamSynchronize( *stream_ ) );
  }
#endif

  // Copy
  if( !src_perm_ptr && !dst_perm_ptr ){
    // No subsetting
    for( ; task + span_task < dims_1; task += span_task ){
      worker_q -> dispatch( [=]{
        for( int i = task * dims_0; i < ( task + span_task ) * dims_0; i++ ){
          dst_ptr[i] = (d)src_ptr[i];
        }
      });
    }

    worker_q -> dispatch( [=]{
      for( int i = task * dims_0; i < dims_1 * dims_0; i++ ){
        dst_ptr[i] = (d)src_ptr[i];
      }
    });

    worker_q -> sync();
  }
  else if( src_perm_ptr && dst_perm_ptr ){
    // Both subsetted
    for( ; task + span_task < dims_1; task += span_task ){
      worker_q -> dispatch( [=]{
        int dst_pos;
        int src_pos;

        for( int i = task; i < task + span_task; i++ ){
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
    }

    worker_q -> dispatch( [=]{
      int dst_pos;
      int src_pos;

      for( int i = task; i < dims_1; i++ ){
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

    worker_q -> sync();
  }
  else if( dst_perm_ptr ){
    // Destination subsetted
    for( ; task + span_task < dims_1; task += span_task ){
      worker_q -> dispatch( [=]{
        int dst_pos;
        int src_pos;

        for( int i = task; i < task + span_task; i++ ){
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
    }

    worker_q -> dispatch( [=]{
      int dst_pos;
      int src_pos;

      for( int i = task; i < dims_1; i++ ){
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

    worker_q -> sync();
  }
  else{
    // Source subsetted
    for( ; task + span_task < dims_1; task += span_task ){
      worker_q -> dispatch( [=]{
        int dst_pos;
        int src_pos;

        for( int i = task; i < task + span_task; i++ ){
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
    }

    worker_q -> dispatch( [=]{
      int dst_pos;
      int src_pos;

      for( int i = task; i < dims_1; i++ ){
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

    worker_q_ptr -> sync();
  }

  if( !worker_q_ptr ){
    delete worker_q;
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
                               sd_queue* worker_q_ptr = NULL,
                               void* stream_ptr       = NULL ){

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

  sd_queue* worker_q = ( !worker_q_ptr ) ? new sd_queue( 4, true ) : worker_q_ptr;

  int span_task = cuR_transfer_task_span( dims_0, dims_1, worker_q );
  int task = 0;

  cudaStream_t* stream_ = (cudaStream_t*) stream_ptr;
  if( stream_ ){
    cudaTry( cudaStreamSynchronize( *stream_ ) );
  }

  // Copy
  if( !src_perm_ptr && !dst_perm_ptr ){
    // No subsetting
    if( stream_ ){
      cudaTry( cudaMemcpyAsync( dst_ptr,
                                src_ptr,
                                sizeof(t) * dims_0 * dims_1,
                                cudaMemcpyHostToDevice,
                                *stream_ ) );
    }else{
      cudaTry( cudaMemcpyAsync( dst_ptr,
                                src_ptr,
                                sizeof(t) * dims_0 * dims_1,
                                cudaMemcpyHostToDevice ) );
    }
  }else if( src_perm_ptr && dst_perm_ptr ){
    // Both subsetted
    for( ; task + span_task < dims_1; task += span_task ){
      worker_q -> dispatch( [=]( void* stream_ptr ){
        int dst_pos;
        int src_pos;

        for( int i = task; i < task + span_task; i++ ){
          if( src_perm_ptr[i] > src_dims_1 ){
            continue;
          }

          if( dst_perm_ptr[i] > dst_dims_1 ){
            continue;
          }

          src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
          dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

          cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                    src_ptr + src_pos,
                                    dims_0 * sizeof(t),
                                    cudaMemcpyHostToDevice,
                                    *(cudaStream_t*)stream_ptr ) );
        }
      });
    }

    worker_q -> dispatch( [=]( void* stream_ptr ){
      int dst_pos;
      int src_pos;

      for( int i = task; i < dims_1; i++ ){
        if( src_perm_ptr[i] > src_dims_1 ){
          continue;
        }

        if( dst_perm_ptr[i] > dst_dims_1 ){
          continue;
        }

        src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
        dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *(cudaStream_t*)stream_ptr ) );
      }
    });

    worker_q -> sync();
  }else if( dst_perm_ptr ){
    for( ; task + span_task < dims_1; task += span_task ){
      worker_q -> dispatch( [=]( void* stream_ptr ){
        int dst_pos;
        int src_pos;

        for( int i = task; i < task + span_task; i++ ){
          if( dst_perm_ptr[i] > dst_dims_1 ){
            continue;
          }

          src_pos = i * dims_0;
          dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

          cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                    src_ptr + src_pos,
                                    dims_0 * sizeof(t),
                                    cudaMemcpyHostToDevice,
                                    *(cudaStream_t*)stream_ptr ) );
        }
      });
    }

    worker_q -> dispatch( [=]( void* stream_ptr ){
      int dst_pos;
      int src_pos;

      for( int i = task; i < dims_1; i++ ){
        if( dst_perm_ptr[i] > dst_dims_1 ){
          continue;
        }

        src_pos = i * dims_0;
        dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *(cudaStream_t*)stream_ptr ) );
      }
    });

    worker_q -> sync();
  }else{
    for( ; task + span_task < dims_1; task += span_task ){
      worker_q -> dispatch( [=]( void* stream_ptr ){
        int dst_pos;
        int src_pos;

        for( int i = task; i < task + span_task; i++ ){
          if( src_perm_ptr[i] > src_dims_1 ){
            continue;
          }

          src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
          dst_pos = i * dims_0;

          cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                    src_ptr + src_pos,
                                    dims_0 * sizeof(t),
                                    cudaMemcpyHostToDevice,
                                    *(cudaStream_t*)stream_ptr ) );
        }
      });
    }

    worker_q -> dispatch( [=]( void* stream_ptr ){
      int dst_pos;
      int src_pos;

      for( int i = task; i < dims_1; i++ ){
        if( src_perm_ptr[i] > src_dims_1 ){
          continue;
        }

        src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
        dst_pos = i * dims_0;

        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyHostToDevice,
                                  *(cudaStream_t*)stream_ptr ) );
      }
    });

    worker_q -> sync();
  }

  if( stream_ ){
    // Also flushes for WDDM
    cudaTry( cudaStreamSynchronize( *stream_ ) );
  }else{
    cudaDeviceSynchronize();
  }

  if( !worker_q_ptr ){
    delete worker_q;
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
                               sd_queue* worker_q_ptr = NULL,
                               void* stream_ptr       = NULL ){

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

  sd_queue* worker_q = ( !worker_q_ptr ) ? new sd_queue( 4, true ) : worker_q_ptr;

  int span_task = cuR_transfer_task_span( dims_0, dims_1, worker_q );
  int task = 0;

  cudaStream_t* stream_ = (cudaStream_t*) stream_ptr;
  if( stream_ ){
    cudaTry( cudaStreamSynchronize( *stream_ ) );
  }

  // Copy
  if( !src_perm_ptr && !dst_perm_ptr ){
    // No subsetting
    if( stream_ ){
      cudaTry( cudaMemcpyAsync( dst_ptr,
                                src_ptr,
                                sizeof(t) * dims_0 * dims_1,
                                cudaMemcpyDeviceToHost,
                                *stream_ ) );
    }else{
      cudaTry( cudaMemcpyAsync( dst_ptr,
                                src_ptr,
                                sizeof(t) * dims_0 * dims_1,
                                cudaMemcpyDeviceToHost ) );
    }
  }else if( src_perm_ptr && dst_perm_ptr ){
    // Both subsetted
    for( ; task + span_task < dims_1; task += span_task ){
      worker_q -> dispatch( [=]( void* stream_ptr ){
        int dst_pos;
        int src_pos;

        for( int i = task; i < task + span_task; i++ ){
          if( src_perm_ptr[i] > src_dims_1 ){
            continue;
          }

          if( dst_perm_ptr[i] > dst_dims_1 ){
            continue;
          }

          src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
          dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

          cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                    src_ptr + src_pos,
                                    dims_0 * sizeof(t),
                                    cudaMemcpyDeviceToHost,
                                    *(cudaStream_t*)stream_ptr ) );
        }
      });
    }

    worker_q -> dispatch( [=]( void* stream_ptr ){
      int dst_pos;
      int src_pos;

      for( int i = task; i < dims_1; i++ ){
        if( src_perm_ptr[i] > src_dims_1 ){
          continue;
        }

        if( dst_perm_ptr[i] > dst_dims_1 ){
          continue;
        }

        src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
        dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyDeviceToHost,
                                  *(cudaStream_t*)stream_ptr ) );
      }
    });

    worker_q -> sync();
  }else if( dst_perm_ptr ){
    for( ; task + span_task < dims_1; task += span_task ){
      worker_q -> dispatch( [=]( void* stream_ptr ){
        int dst_pos;
        int src_pos;

        for( int i = task; i < task + span_task; i++ ){
          if( dst_perm_ptr[i] > dst_dims_1 ){
            continue;
          }

          src_pos = i * dims_0;
          dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

          cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                    src_ptr + src_pos,
                                    dims_0 * sizeof(t),
                                    cudaMemcpyDeviceToHost,
                                    *(cudaStream_t*)stream_ptr ) );
        }
      });
    }

    worker_q -> dispatch( [=]( void* stream_ptr ){
      int dst_pos;
      int src_pos;

      for( int i = task; i < dims_1; i++ ){
        if( dst_perm_ptr[i] > dst_dims_1 ){
          continue;
        }

        src_pos = i * dims_0;
        dst_pos = ( dst_perm_ptr[i] - 1 ) * dims_0;

        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyDeviceToHost,
                                  *(cudaStream_t*)stream_ptr ) );
      }
    });

    worker_q -> sync();
  }else{
    for( ; task + span_task < dims_1; task += span_task ){
      worker_q -> dispatch( [=]( void* stream_ptr ){
        int dst_pos;
        int src_pos;

        for( int i = task; i < task + span_task; i++ ){
          if( src_perm_ptr[i] > src_dims_1 ){
            continue;
          }

          src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
          dst_pos = i * dims_0;

          cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                    src_ptr + src_pos,
                                    dims_0 * sizeof(t),
                                    cudaMemcpyDeviceToHost,
                                    *(cudaStream_t*)stream_ptr ) );
        }
      });
    }

    worker_q -> dispatch( [=]( void* stream_ptr ){
      int dst_pos;
      int src_pos;

      for( int i = task; i < dims_1; i++ ){
        if( src_perm_ptr[i] > src_dims_1 ){
          continue;
        }

        src_pos = ( src_perm_ptr[i] - 1 ) * dims_0;
        dst_pos = i * dims_0;

        cudaTry( cudaMemcpyAsync( dst_ptr + dst_pos,
                                  src_ptr + src_pos,
                                  dims_0 * sizeof(t),
                                  cudaMemcpyDeviceToHost,
                                  *(cudaStream_t*)stream_ptr ) );
      }
    });

    worker_q -> sync();
  }

  if( stream_ ){
    // Also flushes for WDDM
    cudaTry( cudaStreamSynchronize( *stream_ ) );
  }else{
    cudaDeviceSynchronize();
  }

  if( !worker_q_ptr ){
    delete worker_q;
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
                   SEXP src_dims_r,       // Optional*
                   SEXP dst_dims_r,       // Optional**
                   SEXP src_perm_ptr_r,   // Optional*
                   SEXP dst_perm_ptr_r,   // Optional**
                   SEXP src_span_off_r,   // Optional
                   SEXP dst_span_off_r,   // Optional
                   SEXP worker_q_ptr_r,   // Optional
                   SEXP stream_q_ptr_r ){ // Optional

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

  int src_span_off  = Rf_asInteger( src_span_off_r ) - 1;
  int dst_span_off  = Rf_asInteger( dst_span_off_r ) - 1;

  sd_queue* worker_q_ptr = ( R_NilValue == worker_q_ptr_r ) ? NULL :
    (sd_queue*) R_ExternalPtrAddr( worker_q_ptr_r );

  sd_queue* stream_q_ptr = ( R_NilValue == stream_q_ptr_r ) ? NULL :
    (sd_queue*) R_ExternalPtrAddr( stream_q_ptr_r );

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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<double, double>( REAL( src_ptr_r ),
                                                    REAL( dst_ptr_r ),
                                                    dims,
                                                    src_dims,
                                                    dst_dims,
                                                    src_perm_ptr,
                                                    dst_perm_ptr,
                                                    src_span_off,
                                                    dst_span_off,
                                                    worker_q_ptr,
                                                    stream_ptr );
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
                                                  dst_span_off,
                                                  worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                              INTEGER( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              worker_q_ptr,
                                              stream_ptr );
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
                                            dst_span_off,
                                            worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, int>( LOGICAL( src_ptr_r ),
                                              LOGICAL( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              worker_q_ptr,
                                              stream_ptr );
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
                                            dst_span_off,
                                            worker_q_ptr );
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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<double, float>( REAL( src_ptr_r ),
                                                   (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                   dims,
                                                   src_dims,
                                                   dst_dims,
                                                   src_perm_ptr,
                                                   dst_perm_ptr,
                                                   src_span_off,
                                                   dst_span_off,
                                                   worker_q_ptr,
                                                   stream_ptr );
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
                                                 dst_span_off,
                                                 worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              worker_q_ptr,
                                              stream_ptr );
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
                                            dst_span_off,
                                            worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, bool>( LOGICAL( src_ptr_r ),
                                               (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                               dims,
                                               src_dims,
                                               dst_dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off,
                                               worker_q_ptr,
                                               stream_ptr );
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
                                             dst_span_off,
                                             worker_q_ptr );
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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<double, float>( REAL( src_ptr_r ),
                                                   (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                   dims,
                                                   src_dims,
                                                   dst_dims,
                                                   src_perm_ptr,
                                                   dst_perm_ptr,
                                                   src_span_off,
                                                   dst_span_off,
                                                   worker_q_ptr,
                                                   stream_ptr );
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
                                                 dst_span_off,
                                                 worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, int>( INTEGER( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              worker_q_ptr,
                                              stream_ptr );
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
                                            dst_span_off,
                                            worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, bool>( LOGICAL( src_ptr_r ),
                                               (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                               dims,
                                               src_dims,
                                               dst_dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off,
                                               worker_q_ptr,
                                               stream_ptr );
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
                                             dst_span_off,
                                             worker_q_ptr );
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

  case 1:
    switch( dst_level ){
    // 1-0 =====================================================================
    case 0:
      switch( type ){
      case 'n':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<float, double>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                   REAL( dst_ptr_r ),
                                                   dims,
                                                   src_dims,
                                                   dst_dims,
                                                   src_perm_ptr,
                                                   dst_perm_ptr,
                                                   src_span_off,
                                                   dst_span_off,
                                                   worker_q_ptr,
                                                   stream_ptr );
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
                                                 dst_span_off,
                                                 worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              INTEGER( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              worker_q_ptr,
                                              stream_ptr );
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
                                            dst_span_off,
                                            worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<bool, int>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                               LOGICAL( dst_ptr_r ),
                                               dims,
                                               src_dims,
                                               dst_dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off,
                                               worker_q_ptr,
                                               stream_ptr );
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
                                             dst_span_off,
                                             worker_q_ptr );
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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                  (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                  dims,
                                                  src_dims,
                                                  dst_dims,
                                                  src_perm_ptr,
                                                  dst_perm_ptr,
                                                  src_span_off,
                                                  dst_span_off,
                                                  worker_q_ptr,
                                                  stream_ptr );
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
                                                dst_span_off,
                                                worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              worker_q_ptr,
                                              stream_ptr );
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
                                            dst_span_off,
                                            worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                                (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off,
                                                worker_q_ptr,
                                                stream_ptr );
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
                                              dst_span_off,
                                              worker_q_ptr );
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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                  (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                  dims,
                                                  src_dims,
                                                  dst_dims,
                                                  src_perm_ptr,
                                                  dst_perm_ptr,
                                                  src_span_off,
                                                  dst_span_off,
                                                  worker_q_ptr,
                                                  stream_ptr );
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
                                                dst_span_off,
                                                worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              worker_q_ptr,
                                              stream_ptr );
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
                                            dst_span_off,
                                            worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                                (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off,
                                                worker_q_ptr,
                                                stream_ptr );
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
                                              dst_span_off,
                                              worker_q_ptr );
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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_device<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                             (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off,
                                             worker_q_ptr,
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
                                           worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_device<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                           (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           worker_q_ptr,
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
                                         worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_device<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            worker_q_ptr,
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
                                          worker_q_ptr );
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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<float, double>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                   REAL( dst_ptr_r ),
                                                   dims,
                                                   src_dims,
                                                   dst_dims,
                                                   src_perm_ptr,
                                                   dst_perm_ptr,
                                                   src_span_off,
                                                   dst_span_off,
                                                   worker_q_ptr,
                                                   stream_ptr );
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
                                                 dst_span_off,
                                                 worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              INTEGER( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              worker_q_ptr,
                                              stream_ptr );
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
                                            dst_span_off,
                                            worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<bool, int>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                               LOGICAL( dst_ptr_r ),
                                               dims,
                                               src_dims,
                                               dst_dims,
                                               src_perm_ptr,
                                               dst_perm_ptr,
                                               src_span_off,
                                               dst_span_off,
                                               worker_q_ptr,
                                               stream_ptr );
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
                                             dst_span_off,
                                             worker_q_ptr );
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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                  (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                  dims,
                                                  src_dims,
                                                  dst_dims,
                                                  src_perm_ptr,
                                                  dst_perm_ptr,
                                                  src_span_off,
                                                  dst_span_off,
                                                  worker_q_ptr,
                                                  stream_ptr );
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
                                                dst_span_off,
                                                worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              worker_q_ptr,
                                              stream_ptr );
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
                                            dst_span_off,
                                            worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                                (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off,
                                                worker_q_ptr,
                                                stream_ptr );
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
                                              dst_span_off,
                                              worker_q_ptr );
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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<float, float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                                  (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                                  dims,
                                                  src_dims,
                                                  dst_dims,
                                                  src_perm_ptr,
                                                  dst_perm_ptr,
                                                  src_span_off,
                                                  dst_span_off,
                                                  worker_q_ptr,
                                                  stream_ptr );
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
                                                dst_span_off,
                                                worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<int, int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                              (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                              dims,
                                              src_dims,
                                              dst_dims,
                                              src_perm_ptr,
                                              dst_perm_ptr,
                                              src_span_off,
                                              dst_span_off,
                                              worker_q_ptr,
                                              stream_ptr );
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
                                            dst_span_off,
                                            worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_host<bool, bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                                (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                                dims,
                                                src_dims,
                                                dst_dims,
                                                src_perm_ptr,
                                                dst_perm_ptr,
                                                src_span_off,
                                                dst_span_off,
                                                worker_q_ptr,
                                                stream_ptr );
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
                                              dst_span_off,
                                              worker_q_ptr );
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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_device<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                             (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off,
                                             worker_q_ptr,
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
                                           worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_device<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                           (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           worker_q_ptr,
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
                                         worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_host_device<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            worker_q_ptr,
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
                                          worker_q_ptr );
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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_device_host<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                             (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off,
                                             worker_q_ptr,
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
                                           worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_device_host<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                           (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           worker_q_ptr,
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
                                         worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_device_host<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            worker_q_ptr,
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
                                          worker_q_ptr );
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
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_device_host<float>( (float*) R_ExternalPtrAddr( src_ptr_r ),
                                             (float*) R_ExternalPtrAddr( dst_ptr_r ),
                                             dims,
                                             src_dims,
                                             dst_dims,
                                             src_perm_ptr,
                                             dst_perm_ptr,
                                             src_span_off,
                                             dst_span_off,
                                             worker_q_ptr,
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
                                           worker_q_ptr );
        }
        break;

      case 'i':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_device_host<int>( (int*) R_ExternalPtrAddr( src_ptr_r ),
                                           (int*) R_ExternalPtrAddr( dst_ptr_r ),
                                           dims,
                                           src_dims,
                                           dst_dims,
                                           src_perm_ptr,
                                           dst_perm_ptr,
                                           src_span_off,
                                           dst_span_off,
                                           worker_q_ptr,
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
                                         worker_q_ptr );
        }
        break;

      case 'l':
        if( stream_q_ptr ){
          stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
            cuR_transfer_device_host<bool>( (bool*) R_ExternalPtrAddr( src_ptr_r ),
                                            (bool*) R_ExternalPtrAddr( dst_ptr_r ),
                                            dims,
                                            src_dims,
                                            dst_dims,
                                            src_perm_ptr,
                                            dst_perm_ptr,
                                            src_span_off,
                                            dst_span_off,
                                            worker_q_ptr,
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
                                          worker_q_ptr );
        }
        break;

      default:
        Rf_error( "Invalid type in transfer call" );
      }
      break;

      // 3-3 -------------------------------------------------------------------
    case 3:
      if( stream_q_ptr ){
        stream_q_ptr -> dispatch( [=]( void* stream_ptr ){
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
                                         (cudaStream_t*) stream_ptr );
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
                                       dst_span_off );
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
