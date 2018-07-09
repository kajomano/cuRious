template<typename t>
__global__
void cuR_transfer_device_device_cu_kern( t* src_ptr,
                                         t* dst_ptr,
                                         int dims_0,
                                         int dims_1,
                                         int src_dims_1,
                                         int dst_dims_1,
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
  }else if( src_perm_ptr && dst_perm_ptr ){
    // Both subsetted
    for ( int i = index; i < l; i += stride ){
      int src_perm_ind = src_perm_ptr[ i / dims_0 ];
      int src_perm_off = ( src_perm_ind - 1 ) * dims_0;
      int src_off      = i % dims_0;

      if( src_perm_ind > src_dims_1 ){
        // TODO ====
        // Some sort of error message that does not break things
        continue;
      }

      int dst_perm_ind = dst_perm_ptr[ i / dims_0 ];
      int dst_perm_off = ( dst_perm_ind - 1 ) * dims_0;
      int dst_off      = i % dims_0;

      if( dst_perm_ind > dst_dims_1 ){
        // TODO ====
        // Some sort of error message that does not break things
        continue;
      }

      dst_ptr[ dst_perm_off + dst_off ] = src_ptr[ src_perm_off + src_off ];
    }
  }else if( dst_perm_ptr ){
    // Destination subsetted
    for ( int i = index; i < l; i += stride ){
      int dst_perm_ind = dst_perm_ptr[ i / dims_0 ];
      int dst_perm_off = ( dst_perm_ind - 1 ) * dims_0;
      int dst_off      = i % dims_0;

      if( dst_perm_ind > dst_dims_1 ){
        // TODO ====
        // Some sort of error message that does not break things
        continue;
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
        continue;
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
          dims_0,
          dims_1,
          src_dims_1,
          dst_dims_1,
          src_perm_ptr,
          dst_perm_ptr );

      cudaStreamQuery(0);
    }else{
      cuR_transfer_device_device_cu_kern<float><<<numBlocks, blockSize>>>(
          (float*) src_ptr,
          (float*) dst_ptr,
          dims_0,
          dims_1,
          src_dims_1,
          dst_dims_1,
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
          dims_0,
          dims_1,
          src_dims_1,
          dst_dims_1,
          src_perm_ptr,
          dst_perm_ptr );

      cudaStreamQuery(0);
    }else{
      cuR_transfer_device_device_cu_kern<int><<<numBlocks, blockSize>>>(
          (int*) src_ptr,
          (int*) dst_ptr,
          dims_0,
          dims_1,
          src_dims_1,
          dst_dims_1,
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
          dims_0,
          dims_1,
          src_dims_1,
          dst_dims_1,
          src_perm_ptr,
          dst_perm_ptr );

      cudaStreamQuery(0);
    }else{
      cuR_transfer_device_device_cu_kern<bool><<<numBlocks, blockSize>>>(
          (bool*) src_ptr,
          (bool*) dst_ptr,
          dims_0,
          dims_1,
          src_dims_1,
          dst_dims_1,
          src_perm_ptr,
          dst_perm_ptr );

      cudaDeviceSynchronize();
    }
    break;
  }
}
