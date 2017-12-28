#include <cuda.h>
#include <cuda_runtime_api.h>

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

#include "common.h"

#define cudaTry(ans) { cudaError_t code = (ans);                         \
  if( code == cudaSuccess ){                                              \
    Rprintf( "GPU win: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__ );\
  }}
// #define cudaTry(ans) { cudaAssert((ans), __FILE__, __LINE__); }
// inline void cudaAssert( cudaError_t code, char* file, int line ){
//   if( code != cudaSuccess ){
//     //Rprintf( "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line );
//     //exit( code );
//   }else{
//     //Rprintf( "GPU win: %s %s %d\n", cudaGetErrorString(code), file, line );
//   }
// }

extern "C"{
  int get_tensor_length( int n_dims, int* dims );
}
