__global__
void cuR_transfer_device_device_n_cu_kern( float* src,
                                           float* dst,
                                           int dim0,
                                           int dim1,
                                           int* csrc,
                                           int* cdst ){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int l = dim0 * dim1;

  // for (int i = index; i < l; i += stride)
  //   tens_y[i] = tens_x[i] * al + tens_y[i];

  // Copy
  if( !csrc && !cdst ){
    // No subsetting

    // for( int j = 0; j < l; j++ ){
    //   dst[j] = src[j];
    // }
    for ( int i = index; i < l; i += stride ){
      dst[ i ] = src[ i ];
    }
  }else if( csrc && cdst ){
    // Both subsetted
    for ( int i = index; i < l; i += stride ){
      int csrc_ind = ( csrc[ i / dim0 ] - 1 ) * dim0;
      int csrc_off = i % dim0;

      int cdst_ind = ( cdst[ i / dim0 ] - 1 ) * dim0;
      int cdst_off = i % dim0;

      dst[ cdst_ind + cdst_off ] = src[ csrc_ind + csrc_off ];
    }

    // for( int i = 0; i < dim1; i ++ ){
    //   dst_off = (cdst[i]-1)*dim0;
    //   src_off = (csrc[i]-1)*dim0;
    //   for( int j = 0; j < dim0; j++ ){
    //     dst[dst_off+j] = src[src_off+j];
    //   }
    // }
  }else if( !csrc ){
    // Destination subsetted
    for ( int i = index; i < l; i += stride ){
      int cdst_ind = ( cdst[ i / dim0 ] - 1 ) * dim0;
      int cdst_off = i % dim0;

      dst[ cdst_ind + cdst_off ] = src[ i ];
    }

    // int dst_off;
    // int src_off = 0;
    // for( int i = 0; i < dim1; i ++ ){
    //   dst_off = (cdst[i]-1)*dim0;
    //   for( int j = 0; j < dim0; j++ ){
    //     dst[dst_off+j] = src[src_off+j];
    //   }
    //   src_off += dim0;
    // }
  }else{
    // Source subsetted
    for ( int i = index; i < l; i += stride ){
      int csrc_ind = ( csrc[ i / dim0 ] - 1 ) * dim0;
      int csrc_off = i % dim0;

      dst[ i ] = src[ csrc_ind + csrc_off ];
    }

    // int dst_off = 0;
    // int src_off;
    // for( int i = 0; i < dim1; i ++ ){
    //   src_off = (csrc[i]-1)*dim0;
    //   for( int j = 0; j < dim0; j++ ){
    //     dst[dst_off+j] = src[src_off+j];
    //   }
    //   dst_off += dim0;
    // }
  }

}

// void cuR_alg_saxpy_cu_dev( float* tens_x, float* tens_y, int l, float al )
// {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   for (int i = index; i < l; i += stride)
//     tens_y[i] = tens_x[i] * al + tens_y[i];
// }

extern "C"
void cuR_transfer_device_device_n_cu( float* src,
                                      float* dst,
                                      int* dims,
                                      int osrc,
                                      int odst,
                                      int* csrc,
                                      int* cdst,
                                      cudaStream_t* stream ){

  int blockSize = 256;
  int numBlocks = ( ( dims[0]*dims[1] + blockSize ) - 1 ) / blockSize;

  if( osrc ){
    if( csrc ){
      csrc = csrc + osrc;
    }else{
      src  = src + ( osrc * dims[0] );
    }
  }

  if( odst ){
    if( cdst ){
      cdst = cdst + odst;
    }else{
      dst  = dst + ( odst * dims[0] );
    }
  }

  // if( stream ){
  //   cuR_transfer_device_device_n_cu_kern<<<1, 1, 0, *stream>>>( src, dst, dims[0], dims[1], csrc, cdst );
  // }else{
  //   cuR_transfer_device_device_n_cu_kern<<<1, 1>>>( src, dst, dims[0], dims[1], csrc, cdst );
  // }

  if( stream ){
    cuR_transfer_device_device_n_cu_kern<<<numBlocks, blockSize, 0, *stream>>>( src, dst, dims[0], dims[1], csrc, cdst );
  }else{
    cuR_transfer_device_device_n_cu_kern<<<numBlocks, blockSize>>>( src, dst, dims[0], dims[1], csrc, cdst );
  }
}

// void cuR_alg_saxpy_cu(  float* tens_x, float* tens_y, int l, float al, cudaStream_t* stream ){
//   int blockSize = 256;
//   int numBlocks = (l + blockSize - 1) / blockSize;
//
//   if( stream ){
//     cuR_alg_saxpy_cu_dev<<<numBlocks, blockSize, 0, *stream>>>( tens_x, tens_y, l, al );
//   }else{
//     cuR_alg_saxpy_cu_dev<<<numBlocks, blockSize>>>( tens_x, tens_y, l, al );
//   }
// }
