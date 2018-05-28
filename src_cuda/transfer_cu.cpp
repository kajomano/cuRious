#include <stdio.h>

template<typename t>
__global__
void cuR_transfer_device_device_cu_kern( t* src_ptr,
                                         t* dst_ptr,
                                         int src_dims_1,
                                         int dst_dims_1,
                                         int dims_0,
                                         int dims_1,
                                         int* src_perm_ptr,
                                         int* dst_perm_ptr ){

  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int l      = dims_0 * dims_1;

  // Copy
  if( !src_perm_ptr && !dst_perm_ptr ){
    // No subsetting
    for ( int i = index; i < l; i += stride ){
      dst_ptr[ i ] = src_ptr[ i ];
    }
  }else if( !src_perm_ptr && !dst_perm_ptr ){
    // Both subsetted
    for ( int i = index; i < l; i += stride ){
      int src_perm_ind = src_perm_ptr[ i / dims_0 ];
      int src_perm_off = ( src_perm_ind - 1 ) * dims_0;
      int src_off      = i % dims_0;

      if( src_perm_ind > src_dims_1 ){
        // TODO ====
        // Some sort of error message that does not break things
        return;
      }

      int dst_perm_ind = dst_perm_ptr[ i / dims_0 ];
      int dst_perm_off = ( dst_perm_ind - 1 ) * dims_0;
      int dst_off      = i % dims_0;

      if( dst_perm_ind > dst_dims_1 ){
        // TODO ====
        // Some sort of error message that does not break things
        return;
      }

      dst_ptr[ dst_perm_off + dst_off ] = src_ptr[ src_perm_off + src_off ];
    }
  }else if( !src_perm_ptr ){
    // Destination subsetted
    for ( int i = index; i < l; i += stride ){
      int dst_perm_ind = dst_perm_ptr[ i / dims_0 ];
      int dst_perm_off = ( dst_perm_ind - 1 ) * dims_0;
      int dst_off      = i % dims_0;

      if( dst_perm_ind > dst_dims_1 ){
        // TODO ====
        // Some sort of error message that does not break things
        return;
      }

      dst_ptr[ dst_perm_off + dst_off ] = src_ptr[ i ];
    }
  }else{
    // Source subsetted
    for ( int i = index; i < l; i += stride ){
      int src_perm_ind = src_perm_ptr[ i / dims_0 ];
      int src_perm_off = ( src_perm_ind - 1 ) * dims_0;
      int src_off      = i % dims_0;

      if( src_perm_ind > src_dims_1 ){
        // TODO ====
        // Some sort of error message that does not break things
        return;
      }

      dst_ptr[ i ] = src_ptr[ src_perm_off + src_off ];
    }
  }
}

extern "C"
#ifdef _WIN32
__declspec( dllexport )
#endif
  void cuR_transfer_device_device_cu( void* src_ptr,
                                      void* dst_ptr,
                                      const char type,
                                      int* src_dims,
                                      int* dst_dims,
                                      int* dims,
                                      int* src_perm_ptr,
                                      int* dst_perm_ptr,
                                      int src_span_off,
                                      int dst_span_off,
                                      cudaStream_t* stream_ptr ){

    int src_dims_1 = src_dims[1];
    int dst_dims_1 = dst_dims[1];
    int dims_0 = dims[0];
    int dims_1 = dims[1];
    int blockSize = 256;
    int numBlocks = ( ( dims_0 * dims_1 + blockSize ) - 1 ) / blockSize;

    if( src_span_off ){
      if( src_perm_ptr ){
        src_perm_ptr = src_perm_ptr + src_span_off;
      }else{
        switch( type ){
        case 'n':
          src_ptr  = (float*) src_ptr + ( src_span_off * dims_0 );
          break;
        case 'i':
          src_ptr  = (int*) src_ptr + ( src_span_off * dims_0 );
          break;
        case 'l':
          src_ptr  = (bool*) src_ptr + ( src_span_off * dims_0 );
          break;
        }
      }
    }

    if( dst_span_off ){
      if( dst_perm_ptr ){
        dst_perm_ptr = dst_perm_ptr + dst_span_off;
      }else{
        switch( type ){
        case 'n':
          dst_ptr  = (float*) dst_ptr + ( dst_span_off * dims_0 );
          break;
        case 'i':
          dst_ptr  = (int*) dst_ptr + ( dst_span_off * dims_0 );
          break;
        case 'l':
          dst_ptr  = (bool*) dst_ptr + ( dst_span_off * dims_0 );
          break;
        }
      }
    }

  // Kernel calls
  switch( type ){
  case 'n':
    if( stream_ptr ){
      cuR_transfer_device_device_cu_kern<float><<<numBlocks, blockSize, 0, *stream_ptr>>>(
          (float*) src_ptr,
          (float*) dst_ptr,
          src_dims_1,
          dst_dims_1,
          dims_0,
          dims_1,
          src_perm_ptr,
          dst_perm_ptr );
    }else{
      cuR_transfer_device_device_cu_kern<float><<<numBlocks, blockSize>>>(
          (float*) src_ptr,
          (float*) dst_ptr,
          src_dims_1,
          dst_dims_1,
          dims_0,
          dims_1,
          src_perm_ptr,
          dst_perm_ptr );

      cudaDeviceSynchronize();
    }
    break;

  case 'i':
    if( stream_ptr ){
      cuR_transfer_device_device_cu_kern<int><<<numBlocks, blockSize, 0, *stream_ptr>>>(
          (int*) src_ptr,
          (int*) dst_ptr,
          src_dims_1,
          dst_dims_1,
          dims_0,
          dims_1,
          src_perm_ptr,
          dst_perm_ptr );
    }else{
      cuR_transfer_device_device_cu_kern<int><<<numBlocks, blockSize>>>(
          (int*) src_ptr,
          (int*) dst_ptr,
          src_dims_1,
          dst_dims_1,
          dims_0,
          dims_1,
          src_perm_ptr,
          dst_perm_ptr );

      cudaDeviceSynchronize();
    }
    break;

  case 'l':
    if( stream_ptr ){
      cuR_transfer_device_device_cu_kern<bool><<<numBlocks, blockSize, 0, *stream_ptr>>>(
          (bool*) src_ptr,
          (bool*) dst_ptr,
          src_dims_1,
          dst_dims_1,
          dims_0,
          dims_1,
          src_perm_ptr,
          dst_perm_ptr );
    }else{
      cuR_transfer_device_device_cu_kern<bool><<<numBlocks, blockSize>>>(
          (bool*) src_ptr,
          (bool*) dst_ptr,
          src_dims_1,
          dst_dims_1,
          dims_0,
          dims_1,
          src_perm_ptr,
          dst_perm_ptr );

      cudaDeviceSynchronize();
    }
    break;
  }
}

// extern "C"
// #ifdef _WIN32
// __declspec( dllexport )
// #endif
// void cuR_transfer_device_device_i_cu( int* src,
//                                       int* dst,
//                                       int* dims,
//                                       int osrc,
//                                       int odst,
//                                       int* csrc,
//                                       int* cdst,
//                                       cudaStream_t* stream ){
//
//   int blockSize = 256;
//   int numBlocks = ( ( dims[0]*dims[1] + blockSize ) - 1 ) / blockSize;
//
//   if( osrc ){
//     if( csrc ){
//       csrc = csrc + osrc;
//     }else{
//       src  = src + ( osrc * dims[0] );
//     }
//   }
//
//   if( odst ){
//     if( cdst ){
//       cdst = cdst + odst;
//     }else{
//       dst  = dst + ( odst * dims[0] );
//     }
//   }
//
//   if( stream ){
//     cuR_transfer_device_device_n_cu_kern<int><<<numBlocks, blockSize, 0, *stream>>>( src, dst, dims[0], dims[1], csrc, cdst );
//   }else{
//     cuR_transfer_device_device_n_cu_kern<int><<<numBlocks, blockSize>>>( src, dst, dims[0], dims[1], csrc, cdst );
//   }
// }
//
// extern "C"
// #ifdef _WIN32
// __declspec( dllexport )
// #endif
// void cuR_transfer_device_device_l_cu( bool* src,
//                                       bool* dst,
//                                       int* dims,
//                                       int osrc,
//                                       int odst,
//                                       int* csrc,
//                                       int* cdst,
//                                       cudaStream_t* stream ){
//
//   int blockSize = 256;
//   int numBlocks = ( ( dims[0]*dims[1] + blockSize ) - 1 ) / blockSize;
//
//   if( osrc ){
//     if( csrc ){
//       csrc = csrc + osrc;
//     }else{
//       src  = src + ( osrc * dims[0] );
//     }
//   }
//
//   if( odst ){
//     if( cdst ){
//       cdst = cdst + odst;
//     }else{
//       dst  = dst + ( odst * dims[0] );
//     }
//   }
//
//   if( stream ){
//     cuR_transfer_device_device_n_cu_kern<bool><<<numBlocks, blockSize, 0, *stream>>>( src, dst, dims[0], dims[1], csrc, cdst );
//   }else{
//     cuR_transfer_device_device_n_cu_kern<bool><<<numBlocks, blockSize>>>( src, dst, dims[0], dims[1], csrc, cdst );
//   }
// }
