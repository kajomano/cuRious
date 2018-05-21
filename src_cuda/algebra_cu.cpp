__global__
void cuR_alg_saxpy_cu_dev( float* tens_x, float* tens_y, int l, float al )
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < l; i += stride)
    tens_y[i] = tens_x[i] * al + tens_y[i];
}

extern "C"
void cuR_alg_saxpy_cu(  float* tens_x, float* tens_y, int l, float al, cudaStream_t* stream ){
  int blockSize = 256;
  int numBlocks = (l + blockSize - 1) / blockSize;

  if( stream ){
    cuR_alg_saxpy_cu_dev<<<numBlocks, blockSize, 0, *stream>>>( tens_x, tens_y, l, al );
  }else{
    cuR_alg_saxpy_cu_dev<<<numBlocks, blockSize>>>( tens_x, tens_y, l, al );
  }
}
