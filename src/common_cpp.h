#include <cuda.h>
#include <cuda_runtime_api.h>

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

#include "common.h"

#define cudaTry(ans) { cudaAssert((ans), __FILE__, __LINE__); }
void cudaAssert( cudaError_t code, char* file, int line );

int get_tensor_length( int n_dims, int* dims );

