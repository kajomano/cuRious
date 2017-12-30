#include "common_cpp.h"

void cudaAssert( cudaError_t code, char* file, int line ){
  if( code != cudaSuccess ){
    Rprintf( "CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line );
  }
}
