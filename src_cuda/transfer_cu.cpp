template<typename T>
__global__
void cuR_transfer_device_device_n_cu_kern( T* src,
                                           T* dst,
                                           int dim0,
                                           int dim1,
                                           int* csrc,
                                           int* cdst ){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int l = dim0 * dim1;

//   // Copy
//   if( !csrc && !cdst ){
//     // No subsetting
//     for ( int i = index; i < l; i += stride ){
//       dst[ i ] = src[ i ];
//     }
//   }else if( csrc && cdst ){
//     // Both subsetted
//     for ( int i = index; i < l; i += stride ){
//       int csrc_ind = ( csrc[ i / dim0 ] - 1 ) * dim0;
//       int csrc_off = i % dim0;
//
//       int cdst_ind = ( cdst[ i / dim0 ] - 1 ) * dim0;
//       int cdst_off = i % dim0;
//
//       dst[ cdst_ind + cdst_off ] = src[ csrc_ind + csrc_off ];
//     }
//   }else if( !csrc ){
//     // Destination subsetted
//     for ( int i = index; i < l; i += stride ){
//       int cdst_ind = ( cdst[ i / dim0 ] - 1 ) * dim0;
//       int cdst_off = i % dim0;
//
//       dst[ cdst_ind + cdst_off ] = src[ i ];
//     }
//   }else{
//     // Source subsetted
//     for ( int i = index; i < l; i += stride ){
//       int csrc_ind = ( csrc[ i / dim0 ] - 1 ) * dim0;
//       int csrc_off = i % dim0;
//
//       dst[ i ] = src[ csrc_ind + csrc_off ];
//     }
//   }
}

extern "C"
void cuR_transfer_device_device_n_cu( float* src,
                                      float* dst,
                                      int* dims,
                                      int osrc,
                                      int odst,
                                      int* csrc,
                                      int* cdst,
                                      cudaStream_t* stream ){

  int dim0 = dims[0];
  int dim1 = dims[1];
  int blockSize = 256;
  int numBlocks = ( ( dim0*dim1 + blockSize ) - 1 ) / blockSize;

  if( osrc ){
    if( csrc ){
      csrc = csrc + osrc;
    }else{
      src  = src + ( osrc * dim0 );
    }
  }

  if( odst ){
    if( cdst ){
      cdst = cdst + odst;
    }else{
      dst  = dst + ( odst * dim0 );
    }
  }

  // if( stream ){
  //   cuR_transfer_device_device_n_cu_kern<float><<<numBlocks, blockSize, 0, *stream>>>( src, dst, dim0, dim1, csrc, cdst );
  // }else{
    cuR_transfer_device_device_n_cu_kern<float><<<numBlocks, blockSize>>>( src, dst, dim0, dim1, csrc, cdst );
  // }
}

extern "C"
void cuR_transfer_device_device_i_cu( int* src,
                                      int* dst,
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

  if( stream ){
    cuR_transfer_device_device_n_cu_kern<int><<<numBlocks, blockSize, 0, *stream>>>( src, dst, dims[0], dims[1], csrc, cdst );
  }else{
    cuR_transfer_device_device_n_cu_kern<int><<<numBlocks, blockSize>>>( src, dst, dims[0], dims[1], csrc, cdst );
  }
}

extern "C"
void cuR_transfer_device_device_l_cu( bool* src,
                                      bool* dst,
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

  if( stream ){
    cuR_transfer_device_device_n_cu_kern<bool><<<numBlocks, blockSize, 0, *stream>>>( src, dst, dims[0], dims[1], csrc, cdst );
  }else{
    cuR_transfer_device_device_n_cu_kern<bool><<<numBlocks, blockSize>>>( src, dst, dims[0], dims[1], csrc, cdst );
  }
}
