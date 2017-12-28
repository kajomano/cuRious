#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

#include "common.h"

#define cudaTry(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert( cudaError_t code, char* file, int line ){
  if( code != cudaSuccess ){
    printf( "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line );
    //exit( code );
  }else{
    printf( "GPU win: %s %s %d\n", cudaGetErrorString(code), file, line );
  }
}
