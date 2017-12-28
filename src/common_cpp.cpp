#include "common_cpp.h"

int get_tensor_length( int n_dims, int* dims ){
  int l = dims[0];
  for( int i = 1; i < n_dims; i++ ){
    l *= dims[i];
  }

  return l;
}

void cudaAssert( cudaError_t code, char* file, int line ){
  if( code != cudaSuccess ){
    Rprintf( "CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line );
  }
}
